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