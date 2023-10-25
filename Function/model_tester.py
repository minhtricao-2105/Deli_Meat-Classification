import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

class ModelTester():
    """Class for testing a model

    Args:
        model (torch.nn.Module): model to test
        device (str): device to use for testing
        critertion (torch.nn.Module): loss function
        batch_size (int): batch size for testing
        test_dataset (torch.utils.data.Dataset): dataset for testing
    """

    def __init__(self, model, device, criterion, batch_size, test_dataset):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    def evaluate(self):
        self.model.eval()
        self.model.to(self.device)
        total_loss = 0.0
        total_correct = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.test_dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs, labels = inputs.float(), labels.long()
                
                # One-hot encode the labels
                output = self.model(inputs).squeeze()
                loss = self.criterion(output, labels)
                total_loss += loss.item()
                
                # Get the predicted class (not one-hot encoded)
                _, preds = torch.max(output, 1)
                
                total_correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        average_loss = total_loss / len(self.test_dataloader)
        accuracy = total_correct / len(self.test_dataloader.dataset)

        return average_loss, accuracy, all_preds, all_labels
    
    def print_classification_report(self, true_labels, predicted_labels):
        """Print classification report

        Args:
            true_labels (list): true labels
            predicted_labels (list): predicted labels
        """
        result = classification_report(true_labels, predicted_labels)
        print(result)

    def plot_confusion_matrix(self, true_labels, predicted_lables, class_names):
        """ Plot confusion matrix
        
        Args:
            true_labels (list): true labels
            predicted_labels (list): predicted labels
            class_names (list): list of class names
        """
        matrix = confusion_matrix(true_labels, predicted_lables)

        plt.figure(figsize=(10, 7))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

    def one_hot_encode(self, labels, num_classes):
        """One-hot encode the labels."""
        return torch.eye(num_classes, device=labels.device)[labels]


    @classmethod
    def from_config(cls, config):
        try:
            model = config['model']
            criterion = config['criterion']
            device = config['device']
            test_dataset = config['test_dataset']
            batch_size = config['batch_size']
            return cls(model, device, criterion, batch_size, test_dataset)
        except KeyError:
            raise ValueError(f"Config missing member variables for {cls.__name__}")