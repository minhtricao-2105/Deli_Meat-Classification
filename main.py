import os
import multiprocessing
import torch
import torchvision.transforms as transforms
from utils.utils import delete_files_folder, has_folder
from utils.config import load_main_config
from data.dataset import load_deli_meat_csv, split_training_testing_deli_data, DeliMeatDataset, ToTensor
from data.dimensionality_reduction import DimensionReducer
from data.scaling import DataScaler


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


    # split into training and validation data

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

    #model

    # loss and optimizer

    print("starting training...")



