import torch 
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self,input_dim=2,output_dim=1):
        super(LinearRegressionModel,self).__init__()
        self.linear = nn.Linear(input_dim,output_dim)
        
        
    def forward(self,x):
        return self.linear(x)
    