
import Library.Utility as utility
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Model(nn.Module):
    def __init__(self, intention_predictor, xNorm, yNorm):
        super(Model, self).__init__()
        self.IntentionPredictor = intention_predictor
        # self.spatial_encode = torch.nn.Conv1d(153, 1024, 1)
        # self.temporal_encode = torch.nn.Conv1d(6, 32, 1)
        # self.temporal_decode = torch.nn.Conv1d(32, 6, 1)
        # self.spatial_decoode = torch.nn.Conv1d(1024, 51, 1)

        # self.XNorm = xNorm
        # self.YNorm = yNorm

    def forward(self, Input, renormalize=True): 
        #Normalize
        # Input = utility.Normalize(Input, self.XNorm)
        
        # reshape
        # bz = Input.shape[0]
        # char1_past, char1_future, char2_past = Input[:,:306], Input[:,306:612], Input[:,-306:]
        # char1_past_root = char1_past[:,:6*6].reshape([bz,6,-1])
        # char1_past_joint = char1_past[:,6*6:6*6+6*5*9].reshape([bz,6,-1])
        # char1_future_root = char1_future[:,:6*6].reshape([bz,6,-1])
        # char1_future_joint = char1_future[:,6*6:6*6+6*5*9].reshape([bz,6,-1])
        # char2_past_root = char2_past[:,:6*6].reshape([bz,6,-1])
        # char2_past_joint = char2_past[:,6*6:6*6+6*5*9].reshape([bz,6,-1])
        # Input = torch.cat([char1_past_root, char1_past_joint, char1_future_root, char1_future_joint, char2_past_root, char2_past_joint], dim=2)
        
        #Encode X
        Intention = self.IntentionPredictor(Input) # + Input[:,306:612] # residual connection
        # Intention = self.spatial_encode(Input.transpose(1,2))
        # Intention = self.temporal_encode(Intention.transpose(1,2))
        # Intention = self.temporal_decode(Intention)
        # Intention = self.spatial_decoode(Intention.transpose(1,2)).transpose(1,2).reshape([bz,-1])
        
        # Renormalize
        # if renormalize: Intention = utility.Renormalize(Intention, self.YNorm)
        
        return Intention