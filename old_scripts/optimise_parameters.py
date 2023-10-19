#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Script for searching for the optimal classifier parameters. """

import os
import sys
import argparse
import json
import numpy as np
import copy
from itertools import product
from model import load_classifier
import functional as F
from evaluate import *


def classify_grid_search(filename_dict, classifier, preprocess_type, dataset_params, pca_params=None, save=False):
    """ Perform classification for various model parameters.

        Parameters
        ----------
        filename_dict : dict
            Dictionary with the absolute paths to the files storing the training ('train') and testing ('test') data.
        classifier : str
            String identifier for a classification model to find parameters for; if empty or 'all' then find parameters
            for all classifiers.
        preprocess_type : PreprocessType
            Type of preprocessing to be applied to the data (MinMax or Standard scaling).
        dataset_params : dict
            Dictionary of the parameters for the evaluated dataset.
        pca_params : dict
            Dictionary of the Principal Component Analysis parameters.
        save : bool
            True if parameters should be written to file, otherwise False.
    """

    # Load the data
    data_train = load_dataset(filename_dict['train'])
    if args.test == '':
        data_test = data_train
    else:
        data_test = load_dataset(filename_dict['test'])
    data_dict = {'train': data_train, 'test': data_test}

    # Get the training and testing splits
    X_train, X_test, y_train, y_test, _, _ = get_train_test(
        data_dict['train'], preprocess_type, downsample=True, use_validation=True)
    if filename_dict['test'] != filename_dict['train']:
        _, X_test, _, y_test, _, _ = get_train_test(
            data_dict['test'], preprocess_type, downsample=True, use_validation=True)

    print('X_train {}\nX_test {}\ny_train {}\ny_test {}'.format(
        X_train.shape, X_test.shape, y_train.shape, y_test.shape))

    # Reduce dimensionality with PCA
    if pca_params is not None:
        X_train, X_test = transform_pca(
            X_train, X_test, n_components=pca_params['components'], exp_variance=pca_params['variance'])

    # Set up the grid of parameters
    params = get_grid_search_params(classifier)

    # Get unique classifier bases
    unique_bases = set()
    for p in params:
        unique_bases.add(params[p]['base'])
    unique_bases = dict.fromkeys(unique_bases, {'score': 0, 'key': '', 'params': {}})

    # Classification
    results = {}
    for p in params:
        print('{}'.format(params[p]['label']))
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
            # results.append(((p, params[p]['label']), acc, f1, tpr, fpr, roc_auc))
            results[p] = [params[p]['label'], acc, f1, tpr, fpr, roc_auc]

            # Update the best per base
            if acc > unique_bases[params[p]['base']]['score']:
                unique_bases[params[p]['base']] = {'score': acc, 'key': p, 'params': params[p]}

    # Sort the results
    # results = sorted(results, key=lambda x: x[1])[::-1]

    # Print the summary and compose the output json file
    json_file_out = {}
    print('==== BEST PARAMETERS ====')
    for p in unique_bases:
        print(p)
        res = results[unique_bases[p]['key']]
        print('{:20} : Acc - {:.2f}, F1 - {:.2f}, ROC AUC - {:.2f}'.format(
            res[0], res[1] * 100, res[2] * 100, res[5]['micro'] * 100))
        out_params = {key: value for key, value in unique_bases[p]['params'].items() if key not in ['label', 'base']}
        print(out_params)
        json_file_out[unique_bases[p]['params']['base']] = out_params

    # Save
    if save:
        out_filename = './opt_params.json'
        if os.path.exists(out_filename):
            print('File {} already exists, overwrite it? [y/n] '.format(out_filename))
            key_input = str(input()) if sys.version_info[0] >= 3 else str(input())
            if key_input == 'y':
                with open('./opt_params.json', 'w') as fp:
                    json.dump(json_file_out, fp)


def get_grid_search_params(classifier):
    """ Generate a dictionary of model parameters for classification.

        Parameters
        ----------
        classifier : str
            String identifier for a classification model to find parameters for; if empty or 'all' then find parameters
            for all classifiers.

        Returns
        -------
        dict
            Collection of classification model parameters.
    """

    # Set up the grid of parameters
    if classifier == 'all' or classifier == '':
        params = {'svm_linear': {'label': 'SVM LINEAR', 'base': 'svm_linear', 'kernel': 'linear'}}
        params.update(grid_search_svm_rbf())
        params.update(grid_search_svm_poly())
        params.update(grid_search_gp())
        params.update(grid_search_random_forest())
        params.update(grid_search_decision_tree())
        params.update(grid_search_ada_boost())
        params.update(grid_search_knn())
        params.update(grid_search_mlp())
    elif classifier == 'svm_linear':
        params = {'svm_linear': {'label': 'SVM LINEAR', 'base': 'svm_linear', 'kernel': 'linear'}}
    elif classifier == 'svm_rbf':
        params = grid_search_svm_rbf()
    elif classifier == 'svm_polynomial':
        params = grid_search_svm_poly()
    elif classifier == 'gp':
        params = grid_search_gp()
    elif classifier == 'random_forest':
        params = grid_search_random_forest()
    elif classifier == 'decision_tree':
        params = grid_search_decision_tree()
    elif classifier == 'ada_boost':
        params = grid_search_ada_boost()
    elif classifier == 'knn':
        params = grid_search_knn()
    elif classifier == 'mlp':
        params = grid_search_mlp()
    else:
        raise RuntimeError('Unknown classifier type {}'.format(classifier))

    return params


def grid_search_svm_rbf():
    """ Generate a dictionary of parameters for the Radial Basis Function Support Vector Machine classifier.

        Returns
        -------
        dict
            Dictionary of configuration parameters.
    """

    # Set up the grid of parameters
    # base_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    base_range = [0.001, 0.1, 100.0]
    c_2d_range = copy.copy(base_range)
    c_2d_range.extend([i * 5 for i in base_range])
    gamma_2d_range = copy.copy(base_range)
    gamma_2d_range.extend([i * 5 for i in base_range])
    grid_params = {}
    for c in c_2d_range:
        for gamma in gamma_2d_range:
            title = 'svm_rbf_' + str(c) + '_' + str(gamma)
            label = 'SVM RBF, C = ' + str(c) + ', G = ' + str(gamma)
            grid_params[title] = {'label': label, 'base': 'svm_rbf', 'kernel': 'rbf', 'gamma': gamma, 'c': c}

    return grid_params


def grid_search_svm_poly():
    """ Generate a dictionary of parameters for the Polynomial Support Vector Machine classifier.

        Returns
        -------
        dict
            Dictionary of configuration parameters.
    """

    # Set up the grid of parameters
    base_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    c_2d_range = copy.copy(base_range)
    c_2d_range.extend([i * 5 for i in base_range])
    degree_range = [1, 2, 3, 4, 5]
    grid_params = {}
    for c in c_2d_range:
        for deg in degree_range:
            title = 'svm_poly_' + str(c) + '_' + str(deg)
            label = 'SVM POLY, C = ' + str(c) + ', D = ' + str(deg)
            grid_params[title] = {'label': label, 'base': 'svm_poly', 'kernel': 'poly', 'degree': deg, 'c': c}

    return grid_params


def grid_search_gp():
    """ Generate a dictionary of parameters for the Gaussian Process classifier.

        Returns
        -------
        dict
            Dictionary of configuration parameters.
    """

    # Set up the grid of parameters
    # kernels = ['rbf', 'matern', 'rational_quadratic']
    kernels = ['matern']
    length_scale = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    smoothness = [0.5, 1.5, 2.5]  # only for matern
    alpha = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]  # only for rational_quadratic
    n_restarts = [10]  #[0, 5, 10, 20]
    max_iter_predict = [50]  #[50, 100, 250, 500]
    grid_params = {}
    for k in kernels:
        if k == 'rbf':
            args = list(product(length_scale, n_restarts, max_iter_predict))
            for l, n, m in args:
                title = 'gp_' + str(l) + '_' + str(m) + ' ' + str(n)
                label = 'GP, K = RBF, L = ' + str(l) + ', M = ' + str(m) + ', N = ' + str(n)
                grid_params[title] = {'label': label, 'base': 'gp', 'kernel': k, 'length_scale': l,
                                      'n_restarts': n, 'max_iter_predict': m}
        elif k == 'matern':
            args = list(product(length_scale, smoothness, n_restarts, max_iter_predict))
            for l, s, n, m in args:
                title = 'gp_' + str(l) + '_' + str(s) + '_' + str(m) + ' ' + str(n)
                label = 'GP, K = MATERN, L = ' + str(l) + ', S = ' + str(s) + ', M = ' + str(m) + ', N = ' + str(n)
                grid_params[title] = {'label': label, 'base': 'gp', 'kernel': k, 'length_scale': l, 'smoothness': s,
                                      'n_restarts': n, 'max_iter_predict': m}
        elif k == 'rational_quadratic':
            args = list(product(length_scale, alpha, n_restarts, max_iter_predict))
            for l, a, n, m in args:
                title = 'gp_' + str(l) + '_' + str(a) + '_' + str(m) + ' ' + str(n)
                label = 'GP, K = RQ, L = ' + str(l) + ', A = ' + str(a) + ', M = ' + str(m) + ', N = ' + str(n)
                grid_params[title] = {'label': label, 'base': 'gp', 'kernel': k, 'length_scale': l, 'alpha': a,
                                      'n_restarts': n, 'max_iter_predict': m}

    return grid_params


def grid_search_random_forest():
    """ Generate a dictionary of parameters for the Random Forest classifier.

        Returns
        -------
        dict
            Dictionary of configuration parameters.
    """

    # Set up the grid of parameters
    n_estimators_range = [10, 50, 100, 150, 200]
    max_depth_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    grid_params = {}
    for n in n_estimators_range:
        for m in max_depth_range:
            title = 'random_forest_' + str(n) + "_" + str(m)
            label = 'RANDOM FOREST, N = ' + str(n) + ', D = ' + str(m)
            grid_params[title] = {'label': label, 'base': 'random_forest', 'n_estimators': n, 'max_depth': m}

    return grid_params


def grid_search_decision_tree():
    """ Generate a dictionary of parameters for the Decision Tree classifier.

        Returns
        -------
        dict
            Dictionary of configuration parameters.
    """

    # Set up the grid of parameters
    max_depth_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    grid_params = {}
    for m in max_depth_range:
        title = 'decision_tree_' + str(m)
        label = 'DECISION TREE, D = ' + str(m)
        grid_params[title] = {'label': label, 'base': 'decision_tree', 'max_depth': m}

    return grid_params


def grid_search_ada_boost():
    """ Generate a dictionary of parameters for the Ada Boost classifier.

        Returns
        -------
        dict
            Dictionary of configuration parameters.
    """

    # Set up the grid of parameters
    n_estimators_range = [10, 50, 100, 150, 200]
    learning_rate_range = [0.01, 0.1, 1.0, 10.0]
    algorithms = ['SAMME', 'SAMME.R']
    grid_params = {}
    for n in n_estimators_range:
        for l in learning_rate_range:
            for a in algorithms:
                title = 'ada_boost_' + str(n) + '_' + str(l) + '_' + str(a)
                label = 'ADA BOOST, N = ' + str(n) + ', L = ' + str(l) + ', A = ' + str(a)
                grid_params[title] = {'label': label, 'base': 'ada_boost', 'n_estimators': n, 'learning_rate': l,
                                      'algorithm': a}

    return grid_params


def grid_search_knn():
    """ Generate a dictionary of parameters for the K Nearest Neighbours classifier.

        Returns
        -------
        dict
            Dictionary of configuration parameters.
    """

    # Set up the grid of parameters
    n_neighbours_range = [3, 4, 5, 6, 7, 8, 9, 10]
    weights = ['uniform', 'distance']
    p_range = [1, 2]
    grid_params = {}
    for n in n_neighbours_range:
        for w in weights:
            for p in p_range:
                title = 'knn_' + str(n) + '_' + w + '_' + str(p)
                label = 'KNN, N = ' + str(n) + ', W = ' + w + ', P = ' + str(p)
                grid_params[title] = {'label': label, 'base': 'knn', 'n_neighbors': n, 'weights': w, 'p': p}

    return grid_params


def grid_search_mlp():
    """ Generate a dictionary of parameters for the Multi-layer Perceptron classifier.

        Returns
        -------
        dict
            Dictionary of configuration parameters.
    """

    # Set up the grid of parameters
    solver = ['adam', 'sgd']
    hidden_layer_size_range = [(4, ), (8, ), (16, ), (32, ), (4, 4), (8, 4), (16, 8), (32, 16),
                               (4, 4, 4), (8, 8, 4), (16, 8, 4), (32, 16, 8)]
    batch_size = ['auto', 200, 100, 50]
    learning_rate_init = [0.0001, 0.001, 0.01]

    # For SGD
    learning_rate = ['constant', 'invscaling', 'adaptive']
    power_t = [0.1, 0.5, 0.9]
    momentum = [0.1, 0.5, 0.9]
    nesterovs_momentum = [True, False]

    # For ADAM
    beta_1 = [0.1, 0.5, 0.9]
    beta_2 = [0.1, 0.5, 0.9]

    grid_params = {}
    for s in solver:
        if s == 'sgd':
            args = list(product(hidden_layer_size_range, batch_size, learning_rate_init, learning_rate, power_t,
                                momentum, nesterovs_momentum))
            for h, b, li, lr, p, m, n in args:
                title = 'mlp_sgd_' + str(h) + '_' + str(b) + ' ' + str(li) + ' ' + str(lr) + ' ' + str(p) +\
                        ' ' + str(m) + ' ' + str(n)
                label = 'MLP, S = SGD, H = ' + str(h) + ', B = ' + str(b) + ', L = ' + str(li) + ' ' + str(lr) +\
                        ', P = ' + str(p) + ', M = ' + str(m) + ', NM = ' + str(n)
                grid_params[title] = {'label': label, 'base': 'mlp', 'activation': 'relu', 'solver': s,
                                      'hidden_layer_sizes': h, 'batch_size': b, 'learning_rate_init': li,
                                      'learning_rate': lr, 'power_t': p, 'momentum': m, 'nesterovs_momentum': n}
        elif s == 'adam':
            args = list(product(hidden_layer_size_range, batch_size, learning_rate_init, beta_1, beta_2))
            for h, b, li, b1, b2 in args:
                title = 'mlp_adam_' + str(h) + '_' + str(b) + ' ' + str(li) + ' ' + str(b1) + ' ' + str(b2)
                label = 'MLP, S = ADAM, H = ' + str(h) + ", B = " + str(b) + ", L = " + str(li) +\
                        ', B = ' + str(b1) + ' ' + str(b2)
                grid_params[title] = {'label': label, 'base': 'mlp', 'activation': 'relu', 'solver': s,
                                      'hidden_layer_sizes': h, 'batch_size': b, 'learning_rate_init': li, 'beta_1': b1,
                                      'beta_2': b2}

    return grid_params


if __name__ == '__main__':
    # === Parse arguments
    parser = argparse.ArgumentParser(description='Perform a grid search to find best classification parameters')
    parser.add_argument('--root', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--train', type=str)
    parser.add_argument('--test', default='', type=str)
    parser.add_argument('--classifier', default='all', type=str)
    parser.add_argument('--preprocess-type', default='standard', type=str)
    parser.add_argument('--pca', default=-1, type=int)
    parser.add_argument('--save', default=0, type=int)
    args = parser.parse_args()

    # Get the command line input
    train_sub = args.train
    test_sub = train_sub if args.test == '' else args.test
    do_downsample = True
    use_validation = True
    do_save = True if args.save == 1 else False

    # The type of preprocessing
    if args.preprocess_type == 'minmax':
        preprocess = PreprocessType.MINMAX
    else:
        preprocess = PreprocessType.STANDARDISE

    # Parameters
    pca_params = configure_pca(args.pca)
    dataset_params = F.load_dataset_parameters('dataset_parameters.json')
    dataset_params = dataset_params[args.dataset]

    # Create the relevant file names
    input_files = {'train': os.path.join(args.root, args.dataset, train_sub, CSV_FILE_NAME_),
                   'test': os.path.join(args.root, args.dataset, test_sub, CSV_FILE_NAME_),
                   'save': './' + args.dataset + '_' + train_sub + '_' + test_sub}

    # Find the parameters
    classify_grid_search(input_files, args.classifier, preprocess, dataset_params, pca_params=pca_params, save=do_save)
