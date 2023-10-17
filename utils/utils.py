import os
import torch
import numpy as np
import shutil


def get_folder_names(path):
    """Get all folder names in a given directory

    Args:
        path (str): absolute path

    Returns:
        list of str : folder names in path
    """
    folder_names = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    return folder_names



def has_folder(path, folder_name):
    """Checks if path has the given folder

    Args:
        path (str): absolute path
        folder_name (str): folder to check if exists

    Returns:
        bool: is folder exists in path
    """    
    return os.path.exists(os.path.join(path, folder_name))

def clear_gpu_memory(device):
    torch.cuda.empty_cache()


def find_max_batch_size(model, device, input_data_shape, input_labels_shape, dtype=torch.float32, requires_grad=False, safety_tolerance=0.8):
    """
    Find the maximum batch size that can fit on the GPU based on available memory.

    Args:
        model (torch.nn.Module): The PyTorch model.
        device (torch.device): The target device (GPU or CPU).
        dtype (torch.dtype): The data type of the input data (default: torch.float32).

    Returns:
        int: The maximum batch size that can fit on the GPU.
    """

    input_number_pixels = (np.prod(input_data_shape[:-1])).item()

    model_size_bytes = get_model_size(model)

    # Calculate the maximum memory available on the GPU
    max_memory_bytes = torch.cuda.get_device_properties(device).total_memory

    # Dummy input data for memory estimation (requires_grad=False to save memory)
    data = torch.randn(input_data_shape, dtype=dtype, device=device, requires_grad=requires_grad)
    labels = torch.randn(input_labels_shape, dtype=dtype, device=device, requires_grad=requires_grad)

    data_size_bytes = get_tensor_size(data)
    labels_size_bytes = get_tensor_size(labels)

    max_batch_size = np.floor(safety_tolerance*max_memory_bytes/(data_size_bytes + labels_size_bytes + model_size_bytes))
    max_batch_size = (max_batch_size.astype(int)).item()

    max_batch_sizes = {
        'image': max_batch_size,
        'pixel': max_batch_size*input_number_pixels
    }
    
    return max_batch_sizes


def get_model_size(model):
    """
    Get the size of a PyTorch model in bytes.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        int: Size of the model in bytes.
    """
    total_size = 0
    for parameter in model.parameters():
        total_size += parameter.element_size() * parameter.nelement()

    for buffer in model.buffers():
        total_size += buffer.element_size() * buffer.nelement()

    return total_size

def get_tensor_size(input_tensor):
    return input_tensor.element_size() * input_tensor.numel()

def delete_files_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# Usage example:
# max_batch_size = find_max_batch_size(your_model, torch.device("cuda:0"))
# print(f"Maximum Batch Size: {max_batch_size}")