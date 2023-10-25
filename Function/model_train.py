import torch
import torch.nn as nn
import time
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

class ModelTrainer:
    def __init__(self, model, device, train_dataset, val_dataset, batch_size, optimizer, criterion, use_BCE=False):
        """
            The trainer class for training a model in the pytorch framework
        
        Args:
            model (torch.nn.Module): model to train
            device (str): device to use for training
            train_dataset (torch.utils.data.Dataset): dataset for training
            val_dataset (torch.utils.data.Dataset): dataset for validation
            batch_size (int): batch size for training
            optimizer (torch.optim): optimizer for training
            criterion (torch.nn.Module): loss function
            use_BCE (bool, optional): whether to use binary cross entropy loss. Defaults to False.
        """

        self.model = model
        self.device = device
        self.train_dataloder = DataLoader(train_dataset, batch_size=batch_size)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        self.optimizer = optimizer
        self.criterion = criterion
        self.use_BCE = use_BCE
    
    def _train_epoch(self):
        """
        Train model for one epoch
        """
        start_time = time.time()
        self.model.train()
        batch_loss = []
        self.model = self.model.to(self.device)

        for i, (data, labels) in enumerate(self.train_dataloder):
            data, labels = data.to(self.device), labels.to(self.device)
            data, labels = data.float(), labels.long()
            output = self.model(data).squeeze()
            
            # If we implement the BCE Loss, we need to one-hot encode the labels
            if self.use_BCE:
                output = output.view(-1, 4)
                one_hot_labels = self.one_hot_encode(labels, 4)
                loss = self.criterion(output, one_hot_labels.float())
            else:
                loss = self.criterion(output, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_loss.append(loss.item())
        
        train_loss = np.mean(batch_loss)
        epoch_time = time.time() - start_time
        return train_loss, epoch_time
    
    def _validate_epoch(self):
        """
        Validate model for one epoch
        """
        self.model.eval()
        validation_loss = []
        self.model = self.model.to(self.device)

        with torch.no_grad():
            for i, (data, labels) in enumerate(self.val_dataloader):
                data, labels = data.to(self.device), labels.to(self.device)
                data, labels = data.float(), labels.long()
                output = self.model(data).squeeze()
                
                if self.use_BCE:
                    output = output.view(-1, 4)
                    one_hot_labels = self.one_hot_encode(labels, 4)
                    loss = self.criterion(output, one_hot_labels.float())
                else:
                    loss = self.criterion(output, labels)

                validation_loss.append(loss.item())
                

        validation_loss = np.mean(validation_loss)

        return validation_loss

    def one_hot_encode(self, labels, num_classes):
        """
        One-hot encode labels.

        Args:
            labels (torch.Tensor): labels to one-hot encode
            num_classes (int): number of classes in dataset
        """
        return torch.eye(num_classes).to(labels.device)[labels]


    def train(self, num_epochs):
        """
        Train model for num_epochs epochs
        
        Args:
            num_epochs (int): number of epochs to train for
        """
        train_loss_history = []
        val_loss_history = []
        print('Begining training...')

        for epoch in range(num_epochs):
            train_loss, epoch_time = self._train_epoch()
            val_loss = self._validate_epoch()
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            print(f"Epoch: {epoch+1}/{num_epochs} | Train loss: {train_loss:.4f} | Validation loss: {val_loss:.4f} |Epoch time: {epoch_time:.2f}s")

        return train_loss_history, val_loss_history

    def accuracy_fn(self, y_true, y_pred):
        """
        Calculate accuracy of model while training.
        
        Args:
            y_true (torch.Tensor): true labels.
            y_pred (torch.Tensor): predicted labels.
        """
        correct = torch.eq(y_true, y_pred).sum().item()
        accuracy = (correct/len(y_pred))*100
        return accuracy

    @classmethod
    def from_config(cls, config):
        try:
            model = config['model']
            device = config['device']
            train_dataset = config['train_dataset']
            val_dataset = config['val_dataset']
            batch_size = config['batch_size']
            optimizer = config['optimizer']
            criterion = config['criterion']
            use_BCE = config['use_BCE']
            return cls(model, device, train_dataset, val_dataset, batch_size, optimizer, criterion, use_BCE)
        except:
            raise ValueError(f"config missing member variables for {cls.__name__}")