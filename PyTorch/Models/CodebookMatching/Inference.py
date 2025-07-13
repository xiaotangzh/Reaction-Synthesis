
from Library.Utility import *

import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import socket

import Library.Modules as modules
from MotionMatching import *
import Evaluation
import PyTorch.Models.CodebookMatching.model.codebook as EvalModel

start = 10000
initHistory = 50
inputLines = 5000
matchingLines = 0
onChar1Matching = False
onChar2Matching = False
onChar1GTintention = True
evalFrames = -1
restartAt = -1
samples = 10

network, intention_predictor, inputFile, outputFile, GT, Char2MatchingInput, Char2MatchingOutput, Char1MatchingInput, Char1MatchingOutput, BothMatchingInput, BothMatchingOutput = None, None, None, None, None, None, None, None, None, None, None

char1_intention_predictor = torch.load('').eval()

zeroGlobalRoot = torch.zeros(6).cuda()
outputFile = None
inputFileMatching = None
outputFileMatching = None
char1State = None
char1gtjoint, char2gtjoint = None, None
eval_network = None

zeros, ones, noise = torch.zeros(1).cuda(), torch.ones(1).cuda(), torch.ones(1).cuda()*0.5
mse = torch.nn.MSELoss().cuda()

codebook_errors = np.zeros([1000, 1])

def connectUnity3d(network, intention_predictor, inputFile, outputFile, GT, Char2MatchingInput, Char2MatchingOutput, Char1MatchingInput, Char1MatchingOutput, BothMatchingInput, BothMatchingOutput): 
    count = 0
    index = count + start
    minIndex, minIndex1, minIndex2, gtChanged = -1, -1, -1, False
    currents, nexts = None, None
    
    # switch
    MotionMatchingForBoth = False
    DiverseSamples = False
    
    # connect to Unity
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverAddress = ('localhost', 5000)
    serverSocket.bind(serverAddress)
    serverSocket.listen(1)
    print('Waiting for connection ...')
    client, clientAddress = serverSocket.accept()
    print('Successful connection：', clientAddress)
    
    while True:
        receiveBytes = client.recv(50000)
        current = UnserializedToTensor(receiveBytes).unsqueeze(0)
        
        current, current_match, current_char1_intention = current[:,:420+624], current[:,420+624:-270], current[:,-270:]
        current_traj, current_pose = current, current[:,420:]
        
        # restart
        if current.shape[1]==1: 
            count = 0
            index = count + start
            minIndex1, minIndex2 = -1, -1
            currents = torch.zeros([0, inputFile.shape[1]]).float().cuda()
            nexts = torch.zeros([0, outputFile.shape[1]]).float().cuda()
            
        # predict
        if count < 50:
            next = GT[count + start]
            
            #* motion matching for both
            if MotionMatchingForBoth: 
                nextMatch = GT[count + start]
        else:
            #* ground truth
            gt1, gt2 = half0(GT[count + start])
            
            #* produce char 1
            next1 = gt1 # ground truth char1
            # next1, minIndex1 = MotionMatching(Char1MatchingInput, Char1MatchingOutput, current1, minIndex1)
            
            # char1 intention
            char1_intention = char1_intention_predictor(next1[45:45+270].unsqueeze(0))
            next1[45:45+270] = char1_intention[0]
            
            #* produce char 2
            intention = intention_predictor(current_traj)
            
            gt_intention = inputFile[count + start, :126].unsqueeze(0) # use ground turth intention
            gt_current_pose = inputFile[count + start, 126:].unsqueeze(0) # use ground turth pose
            
            next2, codebook_error = network(cat1(intention, gt_current_pose), knn=zeros, gt=inputFile[count + start]) # predict from estimated traj

            if MotionMatchingForBoth and gtChanged:
                next2, codebook_error = network(cat1(intention, current_pose), knn=ones, gt=inputFile[count + start]) # codebook
                next2, minIndex2 = MotionMatching(inputFile[5000:9000], outputFile[5000:9000], cat1(intention, current_pose), minIndex2) 
            
            next = cat0(next1, intention[0], next2)
            
            if DiverseSamples:
                for s in range(1, samples):
                    sample, _ = network(cat1(intention, gt_current_pose), knn=ones)
                    next = cat0(next, sample[-624:])
            
            if MotionMatchingForBoth: 
                if not gtChanged:
                    nextMatch, minIndex = MotionMatching(cat1(BothMatchingInput[:,:100],BothMatchingInput[:,696:800]), BothMatchingOutput, cat1(current_match[:,:100], current_match[:,696:800]), minIndex)
                    if minIndex!= (count + start): 
                        gtChanged = True
                        print(count + start, minIndex, nextMatch[:3])
                else:
                    minIndex += 1
                    nextMatch = BothMatchingOutput[minIndex]
        
        #* motion matching for both
        if MotionMatchingForBoth: next = cat0(next, nextMatch)

        # print(next.shape)
        sendBytes = SerializeFromTensor(next, True if count==0 else False)
        client.sendall(sendBytes)
        
        count += 1
        index += 1
        
        # save codebook errors
        if count> initHistory and count <= (1000+initHistory):
            codebook_errors[count-initHistory-1] = codebook_error.cpu().detach().numpy()
        elif count > (1000+initHistory):
            save_file = "./Evaluation Results/codebook_errors.npy"
            np.save(save_file, codebook_errors)
            print(f"Save file: {save_file}")
            return None
        
        # restart
        if count == restartAt: 
            count = 0
            index = count + start
            minIndex1, minIndex2 = -1, -1
            currents = torch.zeros([0, inputFile.shape[1]]).float().cuda()
            nexts = torch.zeros([0, outputFile.shape[1]]).float().cuda()
        
        # save prediction as .npy for evaluation
        if count == evalFrames:
            np.save(load+"/Eval_matching.npy", currents.detach().cpu().numpy())
            break

def evaluate(pred_network, eval_network, inputFile, outputFile):
    # start from frame 400
    count = evalFrames
    index = count + start
    currents = torch.from_numpy(np.load(load+"/Eval_matching.npy", allow_pickle=True).astype(np.float64)).float().cuda()
    
    evalGT, evalPred, evalOutput = inputFile[index-evalFrames-1+initHistory:index-1], currents, outputFile[index-evalFrames-1+initHistory:index-1],
    
    # generate output based on collected input
    # _, gt, gt_logits = eval_network(evalGT, knn=torch.zeros(1).cuda())
    pred, target_logits, target_probs, target, estimate_logits, estimate_probs, estimate = eval_network(evalPred, t=evalOutput, knn=torch.zeros(1).cuda())
    
    
    # Predict
    pred, target_logits, target_probs, target, estimate_logits, estimate_probs, estimate = pred_network(inputFile, t=outputFile, knn=torch.zeros(1).cuda())

    # Quantitative metrics
    FID(inputFile, outputFile, pred, eval_network)
    MPJPE(outputFile, pred) 
    Diversity(inputFile, outputFile, pred, eval_network)
    MatchingError(target, estimate)
    ErrorOnIntention(outputFile, pred)
    
    # Draw plots
    DrawtSNE_for3(target_logits, estimate_logits)
    DrawCodebookMatchingHeatmap(target, estimate)
    DrawMatchingError()
    DrawLatentSpace()


if __name__ == '__main__':
    dataset = "ReMoCap"
    inputFile = ReadText(f"D:/Exported Data/Reactive Motion Synthesis/{dataset}/matching/Input.txt", maxlines=0, toTorch=True, loadNpy=True)
    outputFile = ReadText(f"D:/Exported Data/Reactive Motion Synthesis/{dataset}/matching/Output.txt", maxlines=0, toTorch=True, loadNpy=True)
    GT = ReadText(f"D:/Exported Data/Reactive Motion Synthesis/{dataset}/ground truth/Output.txt", maxlines=0, toTorch=True, loadNpy=True)
    BothMatchingInput = ReadText(f"D:/Exported Data/Reactive Motion Synthesis/{dataset}/motion matching for both/Input.txt", maxlines=0, toTorch=True, loadNpy=True)
    BothMatchingOutput = ReadText(f"D:/Exported Data/Reactive Motion Synthesis/{dataset}/motion matching for both/Output.txt", maxlines=0, toTorch=True, loadNpy=True)
    
    network = torch.load('').eval()
    intention_predictor = torch.load("").eval()
    
    connectUnity3d(network, intention_predictor, inputFile, outputFile, GT, Char2MatchingInput, Char2MatchingOutput, Char1MatchingInput, Char1MatchingOutput, BothMatchingInput, BothMatchingOutput)
    