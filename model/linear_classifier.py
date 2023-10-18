import torch.nn as nn
import torch.nn.functional as F

class LinearClassifier(nn.Module):
    """Linear classifier with softmax layer

    Args:
        nn (class): inherited from Pytorch
    """    
    def __init__(self, inputSize, outputSize):
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
        """Create instance of model from config dictionary

        Args:
            model_class (class): current model class
            config (dict): dictionary of configs for instantiating model class

        Raises:
            ValueError: Supplied configuration not appropriate for model class

        Returns:
            class: instantiated model class
        """        
        try:
            in_channels = config['input_channels']
            out_channels = config['output_channels']
            instance = model_class(in_channels, out_channels)
            return instance
        except:
            raise ValueError(f"config missing member variables for {model_class.__name__}")