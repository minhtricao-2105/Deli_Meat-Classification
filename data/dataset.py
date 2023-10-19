import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

def load_deli_meat_csv(csv_path):
	"""Loads deli meat classification data from a CSV file

	Args:
		csv_path (str): CSV file path

	Returns:
		dict: Dictionary of deli meat classification pixel data
	"""
	
	data = pd.read_csv(csv_path)

	pixel_class = data.iloc[:, 0].to_numpy().astype(int)
	pixel_train_test_flag = data.iloc[:, 1].to_numpy().astype(bool)
	pixel_location = data.iloc[:, 2:4].to_numpy().astype(int)
	pixel_reflectances = data.iloc[:, 4:].to_numpy()

	deli_meat_data = {
		"pixel_class": pixel_class,
		"pixel_train_test_flag": pixel_train_test_flag,
		"pixel_location": pixel_location,
		"pixel_reflectances": pixel_reflectances
		}

	return deli_meat_data

def split_training_testing_deli_data(deli_meat_data):
	""" Split deli meat data dictionary into training and testing using train_test_flag

	Args:
		deli_meat_data (dict): deli meat data dictionary

	Returns:
		dict: separate deli meat data dictionaries for training and testing
	"""
	pixel_class = deli_meat_data["pixel_class"]
	pixel_train_test_flag = deli_meat_data["pixel_train_test_flag"]
	pixel_location = deli_meat_data["pixel_location"]
	pixel_reflectances = deli_meat_data["pixel_reflectances"]

	# training data
	pixel_class_split = pixel_class[pixel_train_test_flag]
	pixel_location_train_split = pixel_location[pixel_train_test_flag, :]
	pixel_reflectances_split = pixel_reflectances[pixel_train_test_flag, :]

	deli_meat_data_train = {
		"pixel_class": pixel_class_split,
		"pixel_location": pixel_location_train_split,
		"pixel_reflectances": pixel_reflectances_split
		}

	# testing data
	pixel_class_split = pixel_class[~pixel_train_test_flag]
	pixel_location_train_split = pixel_location[~pixel_train_test_flag, :]
	pixel_reflectances_split = pixel_reflectances[~pixel_train_test_flag, :]

	deli_meat_data_test = {
		"pixel_class": pixel_class_split,
		"pixel_location": pixel_location_train_split,
		"pixel_reflectances": pixel_reflectances_split
		}
	
	return deli_meat_data_train, deli_meat_data_test

def split_training_validation_deli_data(deli_meat_data, train_validation_split_ratio):
	"""Splits initial training deli meat dataset into a training and validation datasets. 
	This is done by first picking random pixels for each class, and finding the nearest pixels based on Euclidean distance for each class. 
	The validation dataset for each class should be a collection of nearby pixels. The remaining pixels are used as training data.

	Args:
		deli_meat_data (dict): Initial dictionary of training deli meat data
		train_validation_split_ratio (float): ratio of training:testing pixels (training pixels/total pixels)

	Returns:
		dict: separate deli meat data dictionaries for training and validation
	"""	
	pixel_class = deli_meat_data["pixel_class"]
	pixel_location = deli_meat_data["pixel_location"]
	pixel_reflectances = deli_meat_data["pixel_reflectances"]

	print(f"\nNumber of pixels: {np.size(pixel_class)}")

	# get ID of each class
	unique_classes = np.unique(pixel_class)
	print(f"Unique classes: {unique_classes}")

	# count number of pixels for each class based on split ratio
	pixel_class_count_validation = [
		np.floor((1 - train_validation_split_ratio)*np.size(pixel_class[pixel_class==cls])).astype(int) for cls in unique_classes]
	
	# pick initial random pixel for each class
	initial_indices = []
	for cls in unique_classes:
		initial_indices.append(np.random.choice(np.where(pixel_class == cls)[0]))

	validation_indices = []
	training_indices = []

	# based on split ratio, get sufficient pixels that are near initial pixel for each class for the validation dataset
	for i, idx in enumerate(initial_indices):
		cls = unique_classes[i]
		curr_pt = pixel_location[idx]
		num_pixels = pixel_class_count_validation[i]

		distances = np.linalg.norm(pixel_location[pixel_class == cls, :] - curr_pt, axis=1)
		indices_closest = np.argsort(distances)
		validation_indices = np.concatenate((validation_indices, indices_closest[:num_pixels]))
		training_indices = np.concatenate((training_indices, indices_closest[num_pixels:]))

	print(f"Training pixels: {np.size(training_indices)}, Validation pixels: {np.size(validation_indices)}, Total: {np.size(training_indices) + np.size(validation_indices)}")	

	# return dictionaries for training and validation
	# training data
	pixel_class_split = pixel_class[training_indices]
	pixel_location_train_split = pixel_location[training_indices, :]
	pixel_reflectances_split = pixel_reflectances[training_indices, :]

	deli_meat_data_train = {
		"pixel_class": pixel_class_split,
		"pixel_location": pixel_location_train_split,
		"pixel_reflectances": pixel_reflectances_split
		}

	# validation data
	pixel_class_split = pixel_class[validation_indices]
	pixel_location_train_split = pixel_location[validation_indices, :]
	pixel_reflectances_split = pixel_reflectances[validation_indices, :]

	deli_meat_data_validation = {
		"pixel_class": pixel_class_split,
		"pixel_location": pixel_location_train_split,
		"pixel_reflectances": pixel_reflectances_split
		}
	
	return deli_meat_data_train, deli_meat_data_validation

class DeliMeatDataset(Dataset):
	def __init__(self, deli_meat_data, scaler, reducer, testing=False, transform=None):
		self.labels = deli_meat_data["pixel_class"]
		self.pixel_location = deli_meat_data["pixel_location"]
		self.pixel_reflectances = deli_meat_data["pixel_reflectances"]
		self.transform = transform
		self.number_samples = self.labels.shape[0]

		if testing:
			self.data = scaler.transform(self.pixel_reflectances)
			self.data = reducer.transform(self.data)
		else:
			scaler.fit(self.pixel_reflectances)
			self.data = scaler.transform(self.pixel_reflectances)

			reducer.fit(self.data, self.labels)
			self.data = reducer.transform(self.data)

	def __getitem__(self, index):
		sample = self.data[index,:], np.array(self.labels[index])
		sample = self.transform(sample)
		return sample

	def __len__(self):
		return self.number_samples
	
class ToTensor:
	def __call__(self, sample):
		data, labels = sample
		return torch.from_numpy(data).double(), torch.from_numpy(labels)