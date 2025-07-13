
import Library.Utility as utility
import Library.Plotting as plotting
import Library.AdamWR.adamw as adamw
import Library.AdamWR.cyclic_scheduler as cyclic_scheduler
import Library.Modules as modules

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
import os

import importlib
# from Model import Model

if __name__ == '__main__':
    args = utility.parse_arg()
    Model = importlib.import_module(f"model.{args.model}")
    
    load = os.path.join(args.datapath, "ReMoCap/char1 intention")
    utility.MakeDirectory(f'./checkpoint/{utility.SetupName(args)}')

    XFile = load + "/Input.txt"
    YFile = load + "/Output.txt"
    XFile = utility.ReadText(XFile, args.samples, toTorch=True, loadNpy=True)
    YFile = utility.ReadText(YFile, args.samples, toTorch=True, loadNpy=True)
    XNorm = torch.from_numpy(utility.LoadTxt(load + "/InputNorm.txt", True))
    YNorm = torch.from_numpy(utility.LoadTxt(load + "/OutputNorm.txt", True))
    
    XShape = XFile.shape 
    YShape = YFile.shape 
    Xlabels = load + "/InputLabels.txt"
    Ylabels = load + "/OutputLabels.txt"
    print("Input: "+str(XShape))
    print("Output: "+str(YShape))
    
    sample_count = XShape[0]
    sample_train = int(sample_count*0.95)
    input_dim = XShape[1]
    output_dim = YShape[1]

    utility.SetSeed(23456)

    epochs = args.epochs
    batch_size = args.batch_size
    dropout = args.dropout

    learning_rate = 1e-5
    weight_decay = 1e-5
    restart_period = 10
    restart_mult = 2

    encoder_dim = args.latent_dim
    estimator_dim = args.latent_dim
    decoder_dim = args.latent_dim

    codebook_channels = 128
    codebook_dim = args.codebook_dim
    codebook_size = codebook_channels * codebook_dim
    
    print("Input Features:", input_dim)
    print("Output Features:", output_dim)

    network = utility.ToDevice(Model.Model(
        intention_predictor=modules.LinearEncoder(input_dim, estimator_dim, estimator_dim, output_dim, dropout),
        xNorm=Parameter(XNorm, requires_grad=False),
        yNorm=Parameter(YNorm, requires_grad=False),
    ))    

    optimizer = adamw.AdamW(network.parameters(), lr=learning_rate) 
    loss_function = nn.MSELoss()

    # Training Loop
    I_train = np.arange(sample_train)
    I_test = np.arange(sample_train, sample_count)
    
    for epoch in tqdm(range(epochs)):
        network.train()
        # scheduler.step()
        np.random.shuffle(I_train)
        
        error = 0.0
        error_mse = 0.0
        
        iteration_main = 0
        for i in range(0, sample_train, batch_size):
            train_indices = I_train[i:i+batch_size]
            
            xBatch = XFile[train_indices]
            yBatch = YFile[train_indices]

            prediction = network(xBatch)
            
            mse_loss = loss_function(yBatch, prediction) 
            
            loss = mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.batch_step()

            error += loss.item()
            error_mse += mse_loss.item()

            iteration_main += 1
            
        network.eval()
        np.random.shuffle(I_test)
        xBatch = XFile[I_test]
        yBatch = YFile[I_test]

        prediction = network(xBatch)

        eval_mse_loss = loss_function(yBatch, prediction)
        network.train()
        
        try: torch.save(network, f'./checkpoint/{utility.SetupName(args)}/full_model.pth')
        except: torch.save(network, f'./checkpoint/{utility.SetupName(args)}/full_model2.pth')

        losses = {
                "Intention MSE loss": error_mse/iteration_main,
                "Val Intention MSE loss": eval_mse_loss
            }
        if(args.WandB==1): wandb.log(losses)

