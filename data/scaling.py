import numpy as np
import pickle

class DataScaler:
    def __init__(self, method='normalize', min_range=0.0, max_range=1.0):
        """ Data scaler class

        Args:
            method (str, optional): scaling method to use. Options: 'normalize' or 'standardize. Defaults to 'normalize'.
            min_range (float, optional): if normalization, the minimum range. Defaults to 0.0.
            max_range (float, optional): if normalization, the maximum range. Defaults to 1.0.
        """
        if method in ['normalize', 'standardize']:
            self.method = method
        else:
            raise ValueError(f"{method} Invalid scaling method. Options are 'normalize' or 'standardize'.")

        self.min_range=min_range 
        self.max_range=max_range
        self.min_value=None 
        self.max_value=None
        self.mean = None
        self.std = None
        self.data_fit = False

    @classmethod
    def from_file(cls, file):
        """Constructor for data scaler method previously fit and saved

        Args:
            file (string): file path
        """
        cls.load(file=file)

    def fit(self, data):
        """
        Calculate statistics (mean, std, min, max) based on the input data.

        Args:
            data (numpy.ndarray): The input data to calculate statistics from.
        """
        if self.method == 'normalize':
            self.min_value = np.min(data)
            self.max_value = np.max(data)
        elif self.method == 'standardize':
            self.mean = np.mean(data)
            self.std = np.std(data)
        else:
            raise ValueError("Invalid scaling method. Options are 'normalize' or 'standardize'.")
        
        self.data_fit = True
        
    def transform(self, data):
        """
        Apply the recorded transformation to the input data.

        Args:
            data (numpy.ndarray): The input data to be scaled.

        Returns:
            numpy.ndarray: The scaled data.
        """
        if not self.data_fit:
            self.fit(data)

        if self.method == 'normalize':
            return self.normalize(data)
        elif self.method == 'standardize':
            return self.standardize(data)

    def normalize(self, data):
        """
        Normalize the input data using recorded min and max values between given range. Default range between 0 and 1
        
        Args:
            data (numpy.ndarray): The input data to be normalized.
        Returns:
            numpy.ndarray: The normalized data.
        """        
        normalized_data = self.min_range + (data - self.min_value) * (self.max_range - self.min_range) / (self.max_value - self.min_value)
        normalized_data = np.clip(normalized_data, a_min=self.min_range, a_max=self.max_range)
        return normalized_data

    def standardize(self, data):
        """
        Standardize the input data using the recorded mean and standard deviation.
        
        Args:
            data (numpy.ndarray): The input data to be standardized.
            
        Returns:
            numpy.ndarray: The standardized data.
        """
        standardized_data = (data - self.mean) / self.std
        return standardized_data
    
    def save(self, file):
        """Save fitted scaling model to pickle file

        Args:
            file (str): Full path and filename of pickle file
        """
        with open(file, 'wb') as f:
            pickle.dump((self.min_range, self.max_range, self.min_value, self.max_value, self.mean, self.std, self.data_fit), f)

    def load(self, file):
        """Load fitted scaling model from pickle file

        Args:
            file (str): full path and filename of pickle file
        """
        with open(file, 'rb') as f:
            self.min_range, self.max_range, self.min_value, self.max_value, self.mean, self.std, self.data_fit = pickle.load(f)