
import Library.Utility as utility
import torch
import random
import time

def MotionMatching(inputFile, outputFile, query, lastMinIndex):
    distances = torch.norm(inputFile - query, dim=1)
    distances[max(0, lastMinIndex-100):lastMinIndex+1] = 9999
    minIndex = torch.argmin(distances)
    # while minIndex<=lastMinIndex and abs(minIndex-lastMinIndex)<15:
    #     distances[minIndex] = 999
    #     minIndex = torch.argmin(distances)
    #     print(minIndex)
    #     return outputFile[minIndex], minIndex
    
    # else: 
    # print(lastMinIndex, minIndex, distances[minIndex])
    return outputFile[minIndex], minIndex
    
    
    # 10000frames -> 0.035s/frame -> 28 frame/second

def ExtractPosition(data):
    root, data = data[:,:24], data[:,24:]
    indices = torch.cat([torch.arange(i*9, i*9 + 3) for i in range(52)])
    return torch.cat([root, data[:,indices]], dim=1)

def ExtractChar2Matching(data):
    return torch.cat([data[:,700:752], data[:,932:932+100], data[:,1400:1424], data[:,1604:1604+100]], dim=1)
    
def test():
    load = ""
    InputFile = load + "/Input.txt"
    InputFile = torch.from_numpy(utility.ReadText(InputFile, maxlines=0))

    OutputFile = load + "/Output.txt"
    OutputFile = torch.from_numpy(utility.ReadText(OutputFile, maxlines=0))

    # indices = list(range(len(InputFile)))
    # random.shuffle(indices)
    # for i in indices:
    #     start = time.time()
    #     match = MotionMatching(InputFile, OutputFile, InputFile[i])
    #     print(time.time() - start)