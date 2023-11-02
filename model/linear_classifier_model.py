import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearClassifier(nn.Module):
    def  __init__(self, inputSize, outputSize):
        super().__init__()
        self.model_name = "linear classifier"
        self.layers = nn.Sequential(
            nn.Linear(inputSize, outputSize),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        out = self.layers(x)
        return out
    
    @classmethod
    def from_config(model_class, config):
        try:
            inputSize = config['inputSize']
            outputSize = config['outputSize']
            instance = model_class(inputSize, outputSize)
            return instance
        except:
            raise ValueError(f"config missing member variables for {model_class.__name__}")
        


