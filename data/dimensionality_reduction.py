import numpy as np
import pickle
from sklearn import decomposition
from sklearn.feature_selection import RFE
# from sklearn.svm import SVR

methods_dict = {
    'FA': decomposition.FactorAnalysis,
    'PCA': decomposition.PCA,
    # 'RFE': lambda n_components: RFE(SVR(kernel="linear"), n_features_to_select=n_components, step=1),
    # 'U': Unsupervised,
    # 'S': Supervised,
    # 'M': Manual
}

class DimensionReducer:
    def __init__(self, n_components, reduction_method='FA'):
        self.reduction_method = reduction_method
        self.n_components = n_components
        self.reduction_model = self.get_reduction_model()

    @classmethod
    def from_file(cls, file):
        """Constructor for dimensionality reduction method previously fit and saved

        Args:
            file (string): file path
        """
        cls.load(file=file)

    def fit(self, X, y=None):
        """Fit data to create dimensionality reduction transformation

        Args:
            X (numpy 2d Array): list of reflectance pixels
            y (numpy 1d array, optional): output class. Defaults to None.

        """        
        if y is None:
            self.reduction_model.fit(X)
        else:
            self.reduction_model.fit(X, y)

    def transform(self, X):
        """Apply dimensionality reduction to input data

        Args:
            X (numpy 2d Array): list of reflectance pixels

        Returns:
            (numpy 2d Array): reduced data after applying transformation
        """        
        x_transformed = self.reduction_model.transform(X)
        return x_transformed

    def save(self, file):
        """Save fitted reduction model to pickle file

        Args:
            file (str): Full path and filename of pickle file
        """        
        with open(file, 'wb') as f:
            pickle.dump((self.reduction_model, self.n_components, self.reduction_method), f)

    def load(self, file):
        """Load fitted reduction model from pickle file

        Args:
            file (str): full path and filename of pickle file
        """        
        with open(file, 'rb') as f:
            self.reduction_model, self.n_components, self.reduction_method = pickle.load(f)
    
    def get_reduction_model(self):
        """load dimensionality reduction method from reduction method name in string.

        Raises:
            ValueError: Error if reduction method does not exist

        Returns:
            (reduction method class): reduction method class
        """    
        try:
            return methods_dict[self.reduction_method](self.n_components)
        except:
            raise ValueError(f"Unknown reduction method: {self.reduction_method}")