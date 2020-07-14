import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedNLLLoss(nn.Module):
    """
    Masked NLL Loss
    
    """

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')


    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        
        """

        # batch*seq_len, 1
        mask_ = mask.view(-1,1)
        
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target) / torch.sum(self.weight[target]*mask_.squeeze())
            
        return loss


class MaskedMSELoss(nn.Module):
    """
    Masked MSE Loss
    
    """

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        
        self.loss = nn.MSELoss(reduction='sum')


    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        
        """
        
        loss = self.loss(pred*mask, target) / torch.sum(mask)
        
        return loss


class UnMaskedWeightedNLLLoss(nn.Module):
    """
    Unmasked weighted NLLLoss
    
    """

    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')
        

    def forward(self, pred, target):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        
        """
        
        if type(self.weight) == type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target) / torch.sum(self.weight[target])
            
        return loss
