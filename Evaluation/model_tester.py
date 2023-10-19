import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTester():
    """Class for testing a model

    Args:
        model (class): model to test
        critertion (class): loss function
        device (str): device to run model on
    """

    def __init__(self, model, critertion, device):
        self.model = model
        self.criterion = critertion
        self.device = device
    
    def evaluate(self, x_test, y_test):
        """ Evaluate model on data

        Args:
            x_test (tensor): input data
            y_test (tensor): labels
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
                        
            outputs = self.model(x_test)
            loss = self.criterion(outputs, y_test)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            total_correct += (preds == y_test).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_test.cpu().numpy())

        average_loss = total_loss / len(y_test)
        accuracy = total_correct / len(y_test)

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