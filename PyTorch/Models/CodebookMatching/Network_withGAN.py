
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
    Model_GAN = importlib.import_module(f"model.GAN")
    
    load = os.path.join(args.datapath, "ReMoCap/matching")
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

    learning_rate = 1e-4
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

    #* define model
    network = utility.ToDevice(Model.Model(
        encoder=modules.LinearEncoder(output_dim + input_dim, encoder_dim, encoder_dim, codebook_size, dropout),
        estimator=modules.LinearEncoder(input_dim, estimator_dim, estimator_dim, codebook_size, dropout),
        decoder=modules.LinearEncoder(codebook_size, decoder_dim, decoder_dim, output_dim, 0.0),
        xNorm=Parameter(XNorm, requires_grad=False),
        yNorm=Parameter(YNorm, requires_grad=False),
        codebook_channels=codebook_channels,
        codebook_dim=codebook_dim
    ))
    
    GAN = Model_GAN.GAN(codebook_size).cuda()
    optimizer_GAN = adamw.AdamW(filter(lambda p: p.requires_grad, GAN.parameters()), lr=learning_rate)
    
    optimizer = adamw.AdamW(filter(lambda p: p.requires_grad, network.parameters()), lr=learning_rate) #, weight_decay=weight_decay)
    scheduler = cyclic_scheduler.CyclicLRWithRestarts(optimizer=optimizer, batch_size=batch_size, epoch_size=sample_count, restart_period=restart_period, t_mult=restart_mult, policy="cosine", verbose=True)
    loss_function = nn.MSELoss()

    # Training Loop
    I_train = np.arange(sample_train)
    I_test = np.arange(sample_train, sample_count)
    
    # GAN
    gan_update_batch = 10000
    
    for epoch in tqdm(range(epochs)):
        network.train()
        GAN.train()
        scheduler.step()
        np.random.shuffle(I_train)
        
        error = 0.0
        error_mse = 0.0
        error_matching = 0.0
        error_discriminator = 0.0
        error_generator = 0.0
        
        iteration_main = 0
        iteration_gan = 0
        for i in range(0, sample_train, batch_size):
            train_indices = I_train[i:i+batch_size]
            
            xBatch = XFile[train_indices]
            yBatch = YFile[train_indices]

            prediction, target_logits, target_probs, target, estimate_logits, estimate_probs, estimate = network(
                xBatch, 
                knn=torch.ones(1, device=xBatch.device), 
                t=yBatch,
            )
            
            mse_loss = loss_function(utility.Normalize(yBatch, network.YNorm), utility.Normalize(prediction, network.YNorm)) 
            matching_loss = loss_function(target, estimate) 
            
            # GAN
            gan_loss = utility.DiscriminatorLoss(estimate_logits, target_logits, GAN)[0]
            generator_loss = utility.GeneratorLoss(estimate_logits, target_logits, GAN, 0.0001)[0]
            
            if iteration_main * batch_size < gan_update_batch:
                loss = matching_loss + mse_loss + generator_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.batch_step()
                
                iteration_main += 1
            else:
                loss = gan_loss
                
                optimizer_GAN.zero_grad()
                loss.backward()
                optimizer_GAN.step()
                
                iteration_gan += 1

            error += loss.item()
            error_mse += mse_loss.item()
            error_matching += matching_loss.item()
            error_generator += generator_loss.item()
            error_discriminator += gan_loss.item()
            
        # eval
        network.eval()
        np.random.shuffle(I_test)
        xBatch = XFile[I_test]
        yBatch = YFile[I_test]

        prediction, target_logits, target_probs, target, estimate_logits, estimate_probs, estimate = network(
            xBatch, 
            knn=torch.zeros(1, device=xBatch.device), 
            t=yBatch,
        )

        eval_mse_loss = loss_function(utility.Normalize(yBatch, network.YNorm), utility.Normalize(prediction, network.YNorm)) 
        eval_matching_loss = loss_function(target, estimate) 
        network.train()
        
        try: torch.save(network, f'./checkpoint/{utility.SetupName(args)}/full_model.pth')
        except: torch.save(network, f'./checkpoint/{utility.SetupName(args)}/full_model2.pth')

        losses = {
                "MSE loss": error_mse/iteration_main,
                "Matching loss": error_matching/iteration_main,
                "Generator loss": error_generator/iteration_main,
                "Discriminator loss": error_discriminator/iteration_gan,
                "Eval MSE loss": eval_mse_loss,
                "Eval Matching loss": eval_matching_loss,
            }
        if(args.WandB==1): wandb.log(losses)

