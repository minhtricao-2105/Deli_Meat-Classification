import torch.nn as nn
import torch.nn.functional as F

class LinearClassifier(nn.Module):
    """Linear classifier with softmax layer

    Args:
        nn (class): inherited from Pytorch
    """    
    def __init__(self, inputSize, hiddenSize1, hiddenSize2, outputSize):
        super().__init__()
        self.model_name = "linear classifier"
        
        self.layers = nn.Sequential(
            nn.Linear(inputSize, hiddenSize1),
            nn.BatchNorm1d(hiddenSize1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(hiddenSize1, hiddenSize2),
            nn.BatchNorm1d(hiddenSize2),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(hiddenSize2, outputSize)
        )

    def forward(self, x):
        out = self.layers(x)
        return out
    
    @classmethod
    def from_config(cls, config):
            try:
                inputSize = config['inputSize']
                hiddenSize1 = config['hiddenSize1']
                hiddenSize2 = config['hiddenSize2']
                outputSize = config['outputSize']
                return cls(inputSize, hiddenSize1, hiddenSize2, outputSize)
            except KeyError:
                raise ValueError(f"Config missing member variables for {cls.__name__}")
    
    def summary(self):
        print(self)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")
    