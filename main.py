import os
import multiprocessing
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from utils.utils import delete_files_folder, has_folder
from utils.config import load_main_config
from data.dataset import load_deli_meat_csv, split_training_testing_deli_data, DeliMeatDataset, ToTensor, split_training_validation_deli_data
from data.dimensionality_reduction import DimensionReducer
from data.scaling import DataScaler
from model.linear_classifier import LinearClassifier
from Evaluation.model_tester import ModelTester
import torch.optim as optim
import matplotlib.pyplot as plt

if __name__ == '__main__':

    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    # delete_files_folder(os.path.join(os.getcwd(), "raytune"))

    # parameters from config file
    config = load_main_config()
    split_ratio_training_validation = config["split_ratio_training_validation"]
    path_training = config["path_training"] 
    path_testing = config["path_testing"]
    reflectance_type = config["reflectance_type"]
    result_dir = config["result_dir"]
    num_epochs = config["number_epochs"]
    perform_tune_hyperparameters = config["tune_hyperparameters"]
    project_name = config["project_name"]
    experiment_name = config["experiment_name"]
    scaler_method = config["preprocessing"]["scaler_method"]
    reduction_components = config["preprocessing"]["reduction_components"]
    reduction_method = config["preprocessing"]["reduction_method"]

    training_testing_is_same_dir = (path_training == path_testing)
    path_training_csv = os.path.join(path_training, reflectance_type + ".csv")
    path_testing_csv = os.path.join(path_testing, reflectance_type + ".csv")

    # check if data directories exist 
    if not os.path.exists(path_training_csv):
        raise Exception(f"{path_training} does not have the {reflectance_type}.csv file")

    if not training_testing_is_same_dir:
        if not os.path.exists(path_testing_csv):
            raise Exception(f"{path_testing} does not have the {reflectance_type}.csv file")
        

    # make directory for results
    result_dir = os.path.join(result_dir, project_name, experiment_name)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    # device processing info
    num_processes = multiprocessing.cpu_count()
    print(f"Number of CPU processors: {num_processes}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")

        # Print GPU information
        for i in range(num_gpus):
            gpu = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu.name}, Total Memory: {gpu.total_memory / (1024**3):.2f} GB")
    else:
        print("No GPU available.")

    # split into training and testing data
    if training_testing_is_same_dir:
        deli_meat_data = load_deli_meat_csv(path_training_csv)
        deli_meat_data_train, deli_meat_data_test = split_training_testing_deli_data(deli_meat_data)
    else:
        deli_meat_data = load_deli_meat_csv(path_training_csv)
        deli_meat_data_train = split_training_testing_deli_data(deli_meat_data)[0]

        deli_meat_data = load_deli_meat_csv(path_testing_csv)
        deli_meat_data_test = split_training_testing_deli_data(deli_meat_data)[1]


    # split into training and validation data (an example of how to use the function)
    deli_meat_data_train, deli_meat_data_validation = split_training_validation_deli_data(
        deli_meat_data_train, split_ratio_training_validation)

    # data scaler and dimensionality reduction objects
    scaler = DataScaler(method=scaler_method)
    reducer = DimensionReducer(n_components=reduction_components, reduction_method=reduction_method)

    # Transforms to apply to data
    apply_transform = transforms.Compose([
        ToTensor()
    ])

    training_dataset = DeliMeatDataset(deli_meat_data_train, scaler=scaler, reducer=reducer, transform=apply_transform)
    testing_dataset = DeliMeatDataset(deli_meat_data_train, scaler=scaler, reducer=reducer, testing=True, transform=apply_transform)

    first_data = training_dataset[0]
    data, labels = first_data
    print(type(data), type(labels))

    # data loader
    # Extract data and labels from training_dataset
    x_train = [sample[0] for sample in training_dataset]
    y_train = [sample[1] for sample in training_dataset]

    # Convert lists to tensors
    x_train = torch.stack(x_train)
    y_train = torch.tensor(y_train)
  
    # Extract data and labels from testing_dataset
    x_test = [sample[0] for sample in testing_dataset]
    y_test = [sample[1] for sample in testing_dataset]

    # Convert lists to tensors
    x_test = torch.stack(x_test)
    y_test = torch.tensor(y_test)

    # Move the data and labels to the device
    x_train = x_train.float().to(device)
    y_train = y_train.to(device)
    x_test = x_test.float().to(device)
    y_test = y_test.to(device)

    # print(x_train.shape, y_train.shape)
    print(type(x_train), type(y_train))

    # Visualize first data
    print(f'First data: {x_train[0]}')
    print(f'First label: {y_train[0]}')

    #Define the model
    config = {
        'inputSize': 30,
        'hiddenSize1': 64,
        'hiddenSize2': 32,
        'outputSize': 4
    }

    model = LinearClassifier.from_config(config)
    model.summary()

    # Move the model to the device
    model = model.to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize lists to store losses for each epoch
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(1000): 
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Store training loss
        train_losses.append(loss.item())
        
        # Validate the model
        model.eval()  
        with torch.no_grad(): 
            val_outputs = model(x_test)
            val_loss = criterion(val_outputs, y_test)
            
            # Store validation loss
            val_losses.append(val_loss.item())
                
        #  print loss values at certain epochs for checking
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Train Loss: {loss.item()}, Validation Loss: {val_loss.item()}')
        
        # Set the model back to training mode
        model.train()  
    
    # Plotting
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses over Epochs')
    plt.show()


    # Model Testing:
    tester = ModelTester(model, criterion, device)
    average_loss, accuracy, all_preds, all_labels = tester.evaluate(x_test, y_test)
    print(f"Test Loss: {average_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Print classification report
    tester.print_classification_report(all_labels, all_preds)

    # Plot confusion matrix
    tester.plot_confusion_matrix(all_labels, all_preds, class_names=['Pork', 'Chicken', 'Beef', 'Turkey'])

    # Save Model:
    model_path = os.path.join(result_dir, "model.pth")