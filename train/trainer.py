import torch
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
from model.config_model import load_model_from_config
from torch.utils.data import DataLoader

from ray import tune, air
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
from data.preprocessing import preprocessing_from_config


def train_epoch(model, data_loader, device, optimizer, criterion):
    start_time = time.time()
    model.train()
    batch_loss = []

    for i, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = inputs.float(), labels.float()
        labels = torch.squeeze(labels)
        optimizer.zero_grad()
        output = torch.squeeze(model(inputs))
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
    
    train_loss = np.mean(batch_loss)
    epoch_time = time.time() - start_time
    return train_loss, epoch_time

def valiate_epoch(model, data_loader, device, criterion):
    model.eval()
    validation_loss = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.float()
            output = model(inputs)
            loss = criterion(output,labels)
            validation_loss.append(loss.item())

    validation_loss = np.mean(validation_loss)

    return validation_loss

def train_best_config(config, dataset_training, device, num_epochs=50):
    # general hyperparameters
    batch_size = int(config["batch_size"])
    learning_rate = config["learning_rate"]
    momentum = config["momentum"]

    # load model
    model = load_model_from_config(config).to(device)

    #loss and optimizer
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # data loaders
    train_data_loader = DataLoader(dataset_training, batch_size=batch_size)  

    for epoch in range(num_epochs):
        train_loss, epoch_time = train_epoch(model, train_data_loader, device, optimizer, criterion)
        print("\tEpoch: {:>2}/{:} \t Average Training Loss: {:>3.3f} \t Epoch Time: {:>5.3f}s".format(epoch + 1, num_epochs, train_loss, epoch_time))










