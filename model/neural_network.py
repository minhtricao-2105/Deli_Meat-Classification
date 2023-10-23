import torch.nn as nn
import torch.nn.functional as F

class NNModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NNModel, self).__init__()

        self.model_name = "neural network"

        # Define the layers:
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

        # drop out layer:
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Pass data through fc1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout
        
        # Pass data through fc2
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Dropout
        
        # Pass data through fc3
        x = self.fc3(x)
        
        return x
    
    @classmethod
    def from_config(model_class, config):
        try:
            input_size = config['input_size']
            output_size = config['output_size']
            hidden_size1 = config['hidden_size1']
            hidden_size2 = config['hidden_size2']
            instance = model_class(input_size, hidden_size1, hidden_size2, output_size)
            return instance
        except:
            raise ValueError(f"config missing member variables for {model_class.__name__}")