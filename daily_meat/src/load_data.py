import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('data/calibrated_reflectance_SVM.csv')

print(f'Data before normalize: {data.head()}%')

# Normalize the reflectance values using the Min-Max Scaling: (x - min(x)) / (max(x) - min(x))
data.iloc[:, 4:] = (data.iloc[:, 4:] - data.iloc[:, 4:].min()) / (data.iloc[:, 4:].max() - data.iloc[:, 4:].min())

print(f'Data after normalize: {data.head()}%')

# Split the data into training and testing sets
# X will be the 4th column and onwards, y will be the 1st column: 
X = data.iloc[:, 4:]
y = data.iloc[:, 0]

# Split data into train and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Split the temporary set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Initialize PCA and the X vector for dimensionality reduction
pca = PCA(n_components=30)

# Apply PCA to the training data and transform the training, validation, and test data
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

# Define the model
model = nn.Sequential(
    nn.Linear(30, 64),  # More units in the first hidden layer
    nn.BatchNorm1d(64),  # Batch normalization
    nn.ReLU(),
    nn.Dropout(0.5),  # Dropout for regularization
    nn.Linear(64, 32),  # Additional hidden layer
    nn.BatchNorm1d(32),  # Batch normalization
    nn.ReLU(),
    nn.Dropout(0.5),  # Dropout for regularization
    nn.Linear(32, 4)   
)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize lists to store losses for each epoch
train_losses = []
val_losses = []

# Training loop (simplified)
for epoch in range(1000):  # Number of epochs
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Store training loss
    train_losses.append(loss.item())

    # Store training loss
    train_losses.append(loss.item())
    
    # Validate the model
    model.eval()  
    with torch.no_grad(): 
        val_outputs = model(torch.tensor(X_val_pca, dtype=torch.float))
        val_loss = criterion(val_outputs, torch.tensor(y_val.values, dtype=torch.long))
        
        # Store validation loss
        val_losses.append(val_loss.item())
    
    model.train()  # Set the model back to training mode
    
    # Optionally, print loss values at certain epochs for checking
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Train Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

# Plotting
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Losses over Epochs')
plt.show()

# Convert validation and test data to PyTorch tensors
X_val_tensor = torch.tensor(X_val_pca, dtype=torch.float)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Set the model to evaluation mode
model.eval()

# Validation
with torch.no_grad():
    val_outputs = model(X_val_tensor)
    _, val_preds = torch.max(val_outputs, 1)
    val_correct = (val_preds == y_val_tensor).sum().item()
    val_accuracy = val_correct / y_val_tensor.size(0)
    print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Testing
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, test_preds = torch.max(test_outputs, 1)
    test_correct = (test_preds == y_test_tensor).sum().item()
    test_accuracy = test_correct / y_test_tensor.size(0)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

