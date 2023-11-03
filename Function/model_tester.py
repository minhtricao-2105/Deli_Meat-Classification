import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle

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
        all_probs = []

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.test_dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs, labels = inputs.float(), labels.long()
                
                # One-hot encode the labels
                output = self.model(inputs).squeeze()
                loss = self.criterion(output, labels)
                total_loss += loss.item()
                
                # Get the predicted probabilities for each class:
                probs = torch.nn.functional.softmax(output, dim=1)
                all_probs.extend(probs.cpu().numpy())

                # Get the predicted class (not one-hot encoded)
                _, preds = torch.max(output, 1)
                
                total_correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        average_loss = total_loss / len(self.test_dataloader)
        accuracy = total_correct / len(self.test_dataloader.dataset)

        # Convert all_probs to numpy array:
        all_probs = np.array(all_probs)

        return average_loss, accuracy, all_preds, all_labels, all_probs
    

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


    def plot_precision_recall_curve(self, true_labels, predicted_probs, class_names):
        """ Plot precision-recall curve for each class
        
        Args:
            true_labels (list or np.array): true labels
            predicted_probs (list or np.array): predicted probabilities for each class
            class_names (list): list of class names
        """
        true_labels = np.array(true_labels)
        predicted_probs = np.array(predicted_probs)

        # Binarize the labels for multi-class plot
        true_labels_bin = label_binarize(true_labels, classes=[0, 1, 2, 3])
        print("Shape of true_labels_bin:", true_labels_bin.shape)
        print("Shape of predicted_probs:", predicted_probs.shape)
        
        if len(predicted_probs.shape) == 1:
            predicted_probs = predicted_probs[:, np.newaxis]

        n_classes = true_labels_bin.shape[1]

        if predicted_probs.shape[1] != n_classes:
            raise ValueError("Number of classes in predicted_probs does not match number of classes in true_labels")

        # Compute Precision-Recall and plot curve
        plt.figure(figsize=(10, 7))
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(true_labels_bin[:, i], predicted_probs[:, i])
            plt.plot(recall, precision, lw=2, label=f'Class {class_names[i]}')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.legend(loc="upper right")
        plt.show()


    def plot_roc_curve(self, true_labels, predicted_probs, class_names):
        """ Plot ROC curve for each class
        
        Args:
            true_labels (list): true labels
            predicted_probs (list): predicted probabilities for each class
            class_names (list): list of class names
        """

        # Convert to numpy arrays:
        true_labels = np.array(true_labels)
        predicted_probs = np.array(predicted_probs)

        # Binarize the labels for multi-class plot
        true_labels_bin = label_binarize(true_labels, classes=[0, 1, 2, 3])
        n_classes = true_labels_bin.shape[1]

        print(true_labels_bin.shape, predicted_probs.shape)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], predicted_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(true_labels_bin.ravel(), predicted_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot ROC curve
        plt.figure(figsize=(10, 7))
        plt.plot(fpr["micro"], tpr["micro"],
                label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
                color='deeppink', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:0.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def compute_roc(self):
        """
        Compute ROC curve and ROC area for each class
        """
        self.model.eval()
        self.model.to(self.device)
        true_labels = []
        predicted_probs = []

        with torch.no_grad():
            for i, (input, labels) in enumerate(self.test_dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs, labels = inputs.float(), labels.long()
                output = self.model(inputs).squeeze()

                probs = torch.softmax(output, dim=1)
                true_labels.extend(labels.cpu().numpy())
                predicted_probs.extend(probs.cpu().numpy())
        
        true_labels = np.array(true_labels)
        predicted_probs = np.array(predicted_probs)

        # Binarize the labels for ROC calculation
        true_labels_bin = label_binarize(true_labels, classes=[0, 1, 2, 3])

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(true_labels_bin.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], predicted_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        return fpr, tpr, roc_auc

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