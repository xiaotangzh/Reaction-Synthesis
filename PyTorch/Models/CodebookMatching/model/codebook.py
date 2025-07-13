
import Library.Utility as utility
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, encoder, estimator, decoder, xNorm, yNorm, codebook_channels, codebook_dim):
        super(Model, self).__init__()

        self.Encoder = encoder
        self.Estimator = estimator
        self.Decoder = decoder

        self.XNorm = xNorm
        self.YNorm = yNorm

        self.C = codebook_channels
        self.D = codebook_dim

    def sample_gumbel(self, tensor, scale, eps=1e-20):
        scale = scale.reshape(-1,1,1,1) #This is noise scale between 0 and 1
        noise = torch.rand_like(tensor) - 0.5 #This is random noise between -0.5 and 0.5
        samples = scale * noise + 0.5 #This is noise rescaled between 0 and 1 where 0.5 is default for 0 noise
        return -torch.log(-torch.log(samples + eps) + eps)
    
    def gumbel_softmax_sample(self, logits, temperature, scale):
        y = logits + self.sample_gumbel(logits, scale)
        return F.softmax(y / temperature, dim=-1)
    
    def gumbel_softmax(self, logits, temperature, scale):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature, scale)

        y_soft = y.view(logits.shape)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        y_hard = y_hard.view(logits.shape)

        return y_soft, y_hard

    def sample(self, z, knn):
        z = z.reshape(-1, self.C, self.D)
        z = z.unsqueeze(0).repeat(knn.size(0), 1, 1, 1)
        z_soft, z_hard = self.gumbel_softmax(z, 1.0, knn)
        z_soft = z_soft.reshape(-1, self.C*self.D)
        z_hard = z_hard.reshape(-1, self.C*self.D)
        return z_soft, z_hard
    
    def forward(self, x, knn, t=None, gt=None): #x=input, knn=samples, t=output
        #training
        if t is not None:
            #Normalize
            x = utility.Normalize(x, self.XNorm)
            t = utility.Normalize(t, self.YNorm)

            #Encode Y
            target_logits = self.Encoder(torch.cat((t,x), dim=1))
            # target_logits = self.Encoder(t)
            target_probs, target = self.sample(target_logits, knn)

            #Encode X
            estimate_logits = self.Estimator(x)
            estimate_probs, estimate = self.sample(estimate_logits, knn)

            #Decode
            y = self.Decoder(target)

            #Renormalize
            return utility.Renormalize(y, self.YNorm), target_logits, target_probs, target, estimate_logits, estimate_probs, estimate
                
        #inference
        else:
            #Normalize
            x = utility.Normalize(x, self.XNorm)
            
            #Encode X
            estimate_logits = self.Estimator(x)
            estimate_probs, estimate = self.sample(estimate_logits, knn)

            #Decode
            y = self.Decoder(estimate)
            
            if gt is not None:
                mse = torch.nn.MSELoss().cuda()
                target_logits = self.Estimator(gt)
                target_probs, target = self.sample(target_logits, knn=torch.zeros(1).cuda())
                return utility.Renormalize(y, self.YNorm)[0], mse(target, estimate)

            else:
                return utility.Renormalize(y, self.YNorm)[0], None