#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Script for training and evaluating the performance of classification models. """

import os
import sys
import argparse
import csv
import pickle
import json
import numpy as np
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import cv2
from model import load_classifier
import functional as F

MAX_TRAINING_THRESHOLD_ = 500
NUM_VALIDATION_ = 500
INPUT_LENGTH_SPLIT_ = 148
# CSV_FILE_NAME_ = 'calibrated_reflectance_SVM.csv'
CSV_FILE_NAME_ = 'reflectance_SVM.csv'

class PreprocessType(Enum):
    """ Definition of the two types of preprocessing available from sklearn: MinMaxScaler and StandardScaler """
    MINMAX = 0
    STANDARDISE = 1


def load_dataset(filename, delimiter=','):
    """ Load dataset formatted in a .csv file.

        Parameters
        ----------
        filename : str
            Absolute path to data file.
        delimiter : str
            Type of delimiter that separates data columns; typically space (' ') or comma (',').

        Returns
        -------
        array_like
            Numpy array of the data where each row is a data element and each column a metadata or feature value.
    """

    # Get the extension type
    _, file_extension = os.path.splitext(filename)

    if file_extension != '.csv':
        raise RuntimeError('Input file must be .csv')

    # Read in the data
    dataset = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=delimiter)
        for row in csvreader:
            dataset.append(row)

    return np.asarray(dataset).astype('float')


def load_classification_parameters(filename, classifier=''):
    """ Load all classifier parameters specified in a .json file.

        Each classifier has its own sub-dictionary of parameters.

        Parameters
        ----------
        filename : str
            Absolute path to parameter file.
        classifier : str
            String identifer for a particular parameters to return; if empty or 'all' then return parameters for all
            classifiers.

        Returns
        -------
        dict
            Dictionary of classifier parameters.
    """
    if filename == '' or not os.path.exists(filename):
        print('No file {}'.format(filename))
        print('Loading default parameters')

        c_params = {
            'svm_linear': {'label': 'SVM Linear', 'kernel': 'linear'},
            'svm_polynomial': {'label': 'SVM Polynomial', 'kernel': 'poly', 'degree': 1, 'c': 10.0},
            'svm_rbf': {'label': 'SVM RBF', 'kernel': 'rbf', 'gamma': 0.001, 'c': 50.0},
            'gp': {'label': 'GP', 'kernel': 'matern', 'length_scale': 1.0, 'smoothness': 2.5, 'n_restarts': 10,
            'max_iter_predict': 200},
            'random_forest': {'label': 'Random Forest', 'n_estimators': 50, 'max_depth': 8},
            'decision_tree': {'label': 'Decision Tree', 'max_depth': 10},
            'ada_boost': {'label': 'Ada Boost', 'n_estimators': 200, 'learning_rate': 1.0, 'algorithm': 'SAMME.R'},
            'knn': {'label': 'KNN', 'n_neighbors': 8, 'weights': 'distance', 'p': 2},
            'qda': {'label': 'QDA'},
            'mlp': {'label': 'MLP', 'hidden_layer_sizes': (4, 4, 4), 'solver': 'sgd', 'activation': 'relu',
                   'batch_size': 200, 'learning_rate_init': 0.01, 'learning_rate': 'constant', 'power_t': 0.5,
                   'momentum': 0.9, 'nesterovs_momentum': True}
        }
    else:
        with open(filename) as json_file:
            c_params = json.load(json_file)

    if classifier != 'all' or classifier != '':
        c_params = {classifier: c_params[classifier]}

    return c_params


def configure_pca(num_components):
    """ Create dictionary of the Principal Component Analysis parameters.

        Parameters
        ----------
        num_components : int
            Number of components that the data is to be reduced to; if -1, then PCA is not applied and if 0, then PCA
            is specified by the desired variance of 0.99 instead.

        Returns
        -------
        dict
            Dictionary of Principal Component Analysis parameters.
    """
    if num_components < 0:
        return None
    else:
        if num_components == 0:
            p_params = {'components': None, 'variance': 0.99}
        else:
            p_params = {'components': num_components, 'variance': None}
        return p_params


def get_all_splits(data_samples, preprocess_type, downsample):
    """ Organise the data in an array into the features for the training, testing and validation splits.

        Parameters
        ----------
        data_samples : array_like
            Data in one array, including both metadata and feature values in each row.
        preprocess_type : PreprocessType
            Type of preprocessing to be applied to the data (MinMax or Standard scaling).
        downsample : bool
            True if the training data should be downsampled (faster processing time) or False if all data is used.

        Returns
        -------
        dict
            Dictionary of the data organised into the training, testing and validation splits.
    """

    # | Class (0 or 1) | Training/Testing (0/1) | u    | v    | Reflectance values of 146 bands ... | ...  |
    # | -------------- | ---------------------- | ---- | ---- | ----------------------------------- | ---- |
    # |                |                        |      |      |                                     |      |

    # Split the dataset into train and test
    train_test = data_samples[:, 1].astype('int')
    y = data_samples[:, 0].astype('int')
    uv = data_samples[:, 2:4].astype('int')
    X = data_samples[:, 4:]
    X_train = np.asarray(X[train_test == 0])
    y_train = np.asarray(y[train_test == 0])
    uv_train = np.asarray(uv[train_test == 0])
    X_test = np.asarray(X[train_test == 1])
    y_test = np.asarray(y[train_test == 1])
    uv_test = np.asarray(uv[train_test == 1])

    # Split the test portion into validation and test (validation used for parameter optimisation)
    np.random.seed(101)  # With a fixed seed, this operation is always the same
    rand_idx_val = np.random.choice(X_test.shape[0], NUM_VALIDATION_, replace=False)
    rand_idx_test = list(set(range(0, X_test.shape[0])).difference(set(rand_idx_val)))
    X_val = X_test[rand_idx_val, :]
    y_val = y_test[rand_idx_val]
    uv_val = uv_test[rand_idx_val, :]
    X_test = X_test[rand_idx_test, :]
    y_test = y_test[rand_idx_test]
    uv_test = uv_test[rand_idx_test, :]

    # Down sample the training set
    if downsample:
        rand_idx_train = np.random.choice(X_train.shape[0], MAX_TRAINING_THRESHOLD_, replace=False)
        X_train = X_train[rand_idx_train, :]
        y_train = y_train[rand_idx_train]
        uv_train = uv_train[rand_idx_train, :]

    # Preprocess the data
    X_train, X_test, X_val = preprocess_samples(X_train, X_test, X_val, preprocess_type)

    # Return a dictionary
    ret = {'train': [X_train, y_train, uv_train],
           'test': [X_test, y_test, uv_test],
           'val': [X_val, y_val, uv_val]}

    return ret


def get_train_test(data, preprocess_type, downsample=False, use_validation=False):
    """ Retrieve the training and testing splits in the dataset.

        Parameters
        ----------
        data : array_like
            Data in one array, including both metadata and feature values in each row.
        preprocess_type : PreprocessType
            Type of preprocessing to be applied to the data (MinMax or Standard scaling).
        downsample : bool
            True if the training data should be downsampled (faster processing time) or False if all data is used.
        use_validation : bool
            True if the testing data should be the smaller validation split or False if all testing data is used.

        Returns
        -------
        tuple
            Data organised into the features (X), labels (y) and corresponding image coordinates (uv) for the
            training, testing and validation splits.
    """
    # Organise the data into its splits
    organised_data = get_all_splits(data, preprocess_type, downsample)

    # Extract the training split
    X_train = organised_data['train'][0]
    y_train = organised_data['train'][1]
    uv_train = organised_data['train'][2]

    # Extract the testing split (either test or validation)
    if use_validation:
        test_key = 'val'
    else:
        test_key = 'test'
    X_test = organised_data[test_key][0]
    y_test = organised_data[test_key][1]
    uv_test = organised_data[test_key][2]

    return X_train, X_test, y_train, y_test, uv_train, uv_test


def preprocess_samples(X_train, X_test, X_val, preprocess_type):
    """ Transform the data by applying a scikit-learn scaler.

        Parameters
        ----------
        X_train : array_like
            Numpy array of training features.
        X_test : array_like
            Numpy array of test features.
        X_test : array_like
            Numpy array of validation features.
        preprocess_type : PreprocessType
            Type of preprocessing to be applied to the data (MinMax or Standard scaling).

        Returns
        -------
        tuple
            Training, testing and validation features after transformation by the scikit-learn scaler.
    """

    # Preprocess the data
    if preprocess_type == PreprocessType.MINMAX:
        # Fit on training data
        norm = MinMaxScaler().fit(X_train)
        # Transform training data
        X_train = norm.transform(X_train)
        # Transform testing data
        X_test = norm.transform(X_test)
        # Transform validation data
        X_val = norm.transform(X_val)
    elif preprocess_type == PreprocessType.STANDARDISE:
        # Fit on training data
        scale = StandardScaler().fit(X_train)
        # Transform training data
        X_train = scale.transform(X_train)
        # Transform testing data
        X_test = scale.transform(X_test)
        # Transform validation data
        X_val = scale.transform(X_val)
    else:
        raise RuntimeError('Unknown preprocessing type')

    return X_train, X_test, X_val


def transform_pca(X_train, X_test, n_components=None, exp_variance=0.95):
    """ Reduce the dimensionality of data with Principal Component Analysis.

        Either reduces to a fixed number of components OR reduces to a fixed explained variance.

        Parameters
        ----------
        X_train : array_like
            Numpy array of training features.
        X_test : array_like
            Numpy array of test features.
        n_components : int
            Dimensionality that the data is to be reduced to.
        exp_variance : float
            Desired amount of variance explained by each of the selected components.

        Returns
        -------
        tuple
            Training, testing and validation features after transformation by the scikit-learn scaler.
    """

    # Make an instance of the PCA model and fit the training data
    if n_components is not None:
        pca = PCA(n_components=n_components)
    else:
        pca = PCA(exp_variance)
    pca.fit(X_train)

    # Apply the mapping to the training and testing data
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print('PCA has reduced data from {} dimensions to {} dimensions with explained variance of {}'.format(
            X_train.shape[1], X_train_pca.shape[1], pca.explained_variance_ratio_))

    # Return the data
    return X_train_pca, X_test_pca


def save_to_file(out_filename, results, pixel_results, pca_params):
    """ Save classification results to file.

        Writes two files. One is all results as a pickle file, to be easily loaded and visualised. The other is a .csv
        file that has the class label for each test data sample.

        Parameters
        ----------
        out_filename : str
            Absolute path of the file that results are to be saved to.
        results : array_like
            Array storing all classification results.
        pixel_results : array_like
            Array storing the classification results for each pixel.
        pca_params : dict
            Dictionary of the Principal Component Analysis parameters (used to include in the filename).
    """
    out_filename += '.pkl'
    pixel_prediction_out_filename = out_filename.replace('.pkl', '_pixel_prediction.csv')
    if pca_params is not None:
        if pca_params['components'] is not None:
            out_filename.replace('.pkl', '_pca_n_' + str(pca_params['components']) + '.pkl')
            pixel_prediction_out_filename.replace('.csv', '_pca_n_' + str(pca_params['components']) + '.csv')
        else:
            out_filename.replace('.pkl', '_pca_v_' + str(pca_params['variance']).replace('.', '-') + '.pkl')
            pixel_prediction_out_filename.replace(
                '.csv', '_pca_v_' + str(pca_params['variance']).replace('.', '-') + '.csv')

    # Check if the pickle file already exists
    do_save = True
    if os.path.exists(out_filename):
        do_save = False
        print('File {} already exists, overwrite it? [y/n] '.format(out_filename))
        key_input = str(input())
        if key_input == 'y':
            do_save = True

    # If the file should be saves, write the pickle file
    if do_save:
        print('Saving to {}'.format(out_filename))
        pickle.dump(results, open(out_filename, 'wb'), protocol=2)

    # Check if the .csv file already exists
    do_save = True
    if os.path.exists(pixel_prediction_out_filename):
        do_save = False
        print('File {} already exists, overwrite it? [y/n] '.format(pixel_prediction_out_filename))
        key_input = str(input())
        if key_input == 'y':
            do_save = True

    # If the file should be saves, write the .csv file
    if do_save:
        print('Saving to {}'.format(pixel_prediction_out_filename))
        with open(pixel_prediction_out_filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(pixel_results)


def generate_pixel_prediction_image(rgb_image, label, y_test, uv_test, pred, sample_scores):
    """ Colourise each pixel in the image as correctly or incorrectly classified.

        Parameters
        ----------
        rgb_image : array_like
            RGB image of the test scene where data (pixels) come from.
        label : str
            Name of the classifier.
        y_test : array_like
            Ground truth class labels.
        uv_test : array_like
            Image coordinates of data samples.
        pred : array_like
            Predicted class labels.
        sample_scores : array_like
            Probability associated to the predictions.
    """

    # Access the global variable
    global csv_rows

    if sample_scores.ndim > 1 and sample_scores.shape[1] > 1:
        sample_scores = np.max(sample_scores, axis=1)

    # Normalise the scores
    sample_scores = np.abs(sample_scores)
    sample_scores = np.clip(sample_scores, 0, 5)
    max_abs_score = np.max(sample_scores)
    min_abs_score = np.min(sample_scores)
    sample_scores = (sample_scores - min_abs_score) / (max_abs_score - min_abs_score)

    rgb_prediction = np.copy(rgb_image)
    fp = np.where(pred != y_test)[0]
    fp_pixels = uv_test[fp, :]
    fp_scores = sample_scores[fp]
    for i in range(fp_pixels.shape[0]):
        csv_rows.append([label.replace(' ', '_'), 0, fp_pixels[i, 0], fp_pixels[i, 1]])
        if not np.isnan(fp_scores[i]):
            rgb_prediction[fp_pixels[i, 0], fp_pixels[i, 1], :] = [0, 0, 50 + 205 * fp_scores[i]]

    tp = np.where(pred == y_test)[0]
    tp_pixels = uv_test[tp, :]
    tp_scores = sample_scores[tp]
    for i in range(tp_pixels.shape[0]):
        csv_rows.append([label.replace(' ', '_'), 1, tp_pixels[i, 0], tp_pixels[i, 1]])
        if not np.isnan(tp_scores[i]):
            rgb_prediction[tp_pixels[i, 0], tp_pixels[i, 1], :] = [0, 50 + 205 * tp_scores[i], 0]

    # Add text
    cv2.putText(rgb_prediction, label, (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2, cv2.LINE_AA)

    return rgb_prediction


def classify(filename_dict, params, preprocess_type, dataset_params, pca_params=None, downsample=True,
             use_validation=False, verbose=False, save=False):
    """ Perform classification with all selected models.

        Parameters
        ----------
        filename_dict : dict
            Dictionary with the absolute paths to the files storing the training ('train') and testing ('test') data.
        params : dict
            Dictionary of all classifier parameters.
        preprocess_type : PreprocessType
            Type of preprocessing to be applied to the data (MinMax or Standard scaling).
        dataset_params : dict
            Dictionary of the parameters for the evaluated dataset.
        pca_params : dict
            Dictionary of the Principal Component Analysis parameters.
        downsample : bool
            True if the training data should be downsampled (faster processing time) or False if all data is used.
        use_validation : bool
            True if the testing data should be the smaller validation split or False if all testing data is used.
        verbose : bool
            True if information about training and testing should be printed to terminal or False if only the final
            results are printed.
        save : bool
            True if classification results should be written to file, otherwise False.

        Returns
        -------
        list
            Classification results for each classifier as tuples containing parameters, name, accuracy, f1 score,
            true positive rate, false positive rate and ROC AUC.
    """

    # Load the data
    data_train = load_dataset(filename_dict['train'])
    if args.test == '':
        data_test = data_train
    else:
        data_test = load_dataset(filename_dict['test'])
    data_dict = {'train': data_train, 'test': data_test}

    # Get the training and testing splits
    X_train, X_test, y_train, y_test, uv_train, uv_test = get_train_test(
        data_dict['train'], preprocess_type, downsample=downsample, use_validation=use_validation)
    if filename_dict['test'] != filename_dict['train']:
        _, X_test, _, y_test, _, uv_test = get_train_test(
            data_dict['test'], preprocess_type, downsample=downsample, use_validation=use_validation)

    print('X_train {}\nX_test {}\ny_train {}\ny_test {}'.format(
        X_train.shape, X_test.shape, y_train.shape, y_test.shape))

    # Reduce dimensionality with PCA
    if pca_params is not None:
        X_train, X_test = transform_pca(
            X_train, X_test, n_components=pca_params['components'], exp_variance=pca_params['variance'])

    # Load the rgb frame if it's available
    image_filename = os.path.join(os.path.dirname(os.path.abspath(filename_dict['test'])), 'false_image.png')
    rgb_frame = cv2.imread(image_filename)

    # Classification
    results = []
    global csv_rows
    csv_rows = []
    image_stack = None
    for p in params:
        print('====================')
        print('== Model Type : {}'.format(params[p]['label']))
        # Retrieve the classifier
        try:
            model = load_classifier(p, params)
        except RuntimeError as e:
            print(e)
            print('Error encountered for {}, skipping'.format(params[p]['label']))
            model = None

        # Make predictions are score them
        if model is not None:
            # Compute accuracy and f1 score
            pred, sample_scores, acc, f1 = F.compute_accuracy(
                model, X_train, y_train, X_test, y_test, dataset_params['classes'])

            # Compute ROC/AUC
            try:
                tpr, fpr, roc_auc = F.compute_roc(model, y_test, sample_scores, dataset_params['classes'])
            except:
                print('Error encountered for {}, skipping'.format(params[p]['label']))
                continue

            # Add the results to the dictionary
            results.append(((p, params[p]['label']), acc, f1, tpr, fpr, roc_auc))

            # Display the image with false positives highlighted in red
            if rgb_frame is not None and uv_test is not None:
                prediction_image = generate_pixel_prediction_image(
                    rgb_frame, params[p]['label'], y_test, uv_test, pred, sample_scores)
                if image_stack is None:
                    image_stack = np.hstack((rgb_frame, prediction_image))
                else:
                    image_stack = np.hstack((image_stack, prediction_image))

            # Print information
            if verbose:
                F.print_scores(params[p]['label'], y_test, pred, acc, f1, roc_auc)
                print('\n')

    # Save
    if save:
        save_to_file(filename_dict['save'], results, csv_rows, pca_params)

    # Print the summary
    if verbose:
        print('==== SUMMARY ====')
        F.print_summary(results)

    # Plot the curves
    if len(results) <= len(F.CLASSIFIER_ORDER_):
        F.plot_roc(results, dataset_params)

    # Visualise the true and false positive pixels on the false image
    if rgb_frame is not None and uv_test is not None:
        cv2.imshow('Pixel prediction', image_stack)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results


if __name__ == '__main__':
    # === Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate classification models')
    parser.add_argument('--root', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--train', type=str)
    parser.add_argument('--test', default='', type=str)
    parser.add_argument('--classifier', default='all', type=str)
    parser.add_argument('--preprocess-type', default='standard', type=str)
    parser.add_argument('--pca', default=-1, type=int)
    parser.add_argument('--down-sample', default=0, type=int)
    parser.add_argument('--use-validation', default=0, type=int)
    parser.add_argument('--save', default=0, type=int)
    args = parser.parse_args()

    # Get the command line input
    train_sub = args.train
    test_sub = train_sub if args.test == '' else args.test
    do_downsample = True if args.down_sample == 1 else False
    use_validation = True if args.use_validation == 1 else False
    do_save = True if args.save == 1 else False

    # The type of preprocessing
    if args.preprocess_type == 'minmax':
        preprocess = PreprocessType.MINMAX
    else:
        preprocess = PreprocessType.STANDARDISE

    # Parameters
    classifier_params = load_classification_parameters('classifier_parameters.json', args.classifier)
    pca_params = configure_pca(args.pca)
    dataset_params = F.load_dataset_parameters('dataset_parameters.json')
    dataset_params = dataset_params[args.dataset]

    # Create the relevant file names
    input_files = {'train': os.path.join(args.root, args.dataset, train_sub, CSV_FILE_NAME_),
                   'test': os.path.join(args.root, args.dataset, test_sub, CSV_FILE_NAME_),
                   'save': './' + args.dataset + '_' + train_sub + '_' + test_sub}

    # Do the classification
    classify(input_files, classifier_params, preprocess, dataset_params, pca_params=pca_params,
             downsample=do_downsample, use_validation=use_validation, verbose=True, save=do_save)
