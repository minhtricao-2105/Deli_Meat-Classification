#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Functions for loading various classification models. """

import sklearn
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def load_classifier(label, params):
    """ Instantiate and return a classifier.

        Parameters
        ----------
        params : dict
            Dictionary of configuration parameters.

        Returns
        -------
        scikit-learn classifier
            Classifier defined in scikit-learn.

        Raises
        ------
        RuntimeError
            If the kernel specified in the configuration is unknown.
    """

    # Store the version of the scikit-learn package
    skl_version = sklearn.__version__.split('.')
    skl_version = float(skl_version[0] + '.' + skl_version[1])

    # Get the classification model
    if 'svm' in label:
        model = _classifier_svm(params[label])
    elif 'gp' in label:
        if skl_version > 0.17:
            model = _classifier_gp(params[label])
        else:
            raise RuntimeError('scikit-learn version {} does not have GaussianProcessClassifier'.format(
                sklearn.__version__))
    elif 'random_forest' in label:
        model = _classifier_rforest(params[label])
    elif 'decision_tree' in label:
        model = _classifier_dtree(params[label])
    elif 'ada_boost' in label:
        model = _classifier_adaboost(params[label])
    elif 'knn' in label:
        model = _classifier_knn(params[label])
    elif 'qda' in label:
        model = _classifier_qda()
    elif 'mlp' in label:
        if skl_version > 0.17:
            model = _classifier_mlp(params[label])
        else:
            raise RuntimeError('scikit-learn version {} does not have MLPClassifier'.format(sklearn.__version__))
    else:
        raise RuntimeError('Unknown model type {}'.format(label))

    return model


def _classifier_svm(params):
    """ Instantiate and return Support Vector Machine classifier.

        Parameters
        ----------
        params : dict
            Dictionary of configuration parameters.

        Returns
        -------
        scikit-learn classifier
            Support Vector Machine classifier defined in scikit-learn.

        Raises
        ------
        RuntimeError
            If the kernel specified in the configuration is unknown.
    """
    if params['kernel'] == 'linear':
        svm_model = svm.SVC(
            kernel='linear'
        )
    elif params['kernel'] == 'poly':
        svm_model = svm.SVC(
            kernel='poly', degree=params['degree'], C=params['c']
        )
    elif params['kernel'] == 'rbf':
        svm_model = svm.SVC(
            kernel='rbf', gamma=params['gamma'], C=params['c']
        )
    else:
        raise RuntimeError('Unknown SVM kernel {}'.format(params['kernel']))

    return svm_model


def _classifier_gp(params):
    """ Instantiate and return Gaussian Process classifier.

        Parameters
        ----------
        params : dict
            Dictionary of configuration parameters.

        Returns
        -------
        scikit-learn classifier
            Gaussian Process classifier defined in scikit-learn.

        Raises
        ------
        RuntimeError
            If the kernel specified in the configuration is unknown.
    """
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic

    if params['kernel'] == 'rbf':
        gp_model = GaussianProcessClassifier(
            kernel=RBF(length_scale=params['length_scale']),
            n_restarts_optimizer=params['n_restarts'], max_iter_predict=params['max_iter_predict'],
            warm_start=False, multi_class='one_vs_rest', random_state=1, n_jobs=-1)
    elif params['kernel'] == 'matern':
        gp_model = GaussianProcessClassifier(
            kernel=Matern(length_scale=params['length_scale'], nu=params['smoothness']),
            n_restarts_optimizer=params['n_restarts'], max_iter_predict=params['max_iter_predict'],
            warm_start=False, multi_class='one_vs_rest', random_state=1, n_jobs=-1)
    elif params['kernel'] == 'rational_quadratic':
        gp_model = GaussianProcessClassifier(
            kernel=RationalQuadratic(length_scale=params['length_scale'], alpha=params['alpha']),
            n_restarts_optimizer=params['n_restarts'], max_iter_predict=params['max_iter_predict'],
            warm_start=False, multi_class='one_vs_rest', random_state=1, n_jobs=-1)
    else:
        raise RuntimeError('Unknown GP kernel {}'.format(params['kernel']))

    return gp_model


def _classifier_rforest(params):
    """ Instantiate and return Random Forest classifier.

        Parameters
        ----------
        params : dict
            Dictionary of configuration parameters.

        Returns
        -------
        scikit-learn classifier
            Random Forest classifier model defined in scikit-learn.
    """
    rforest_model = RandomForestClassifier(
        n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=1
    )

    return rforest_model


def _classifier_dtree(params):
    """ Instantiate and return Decision Tree classifier.

        Parameters
        ----------
        params : dict
            Dictionary of configuration parameters.

        Returns
        -------
        scikit-learn classifier
            Decision Tree classifier defined in scikit-learn.
    """
    dtree_model = DecisionTreeClassifier(
        max_depth=params['max_depth']
    )

    return dtree_model


def _classifier_adaboost(params):
    """ Instantiate and return Ada Boost classifier.

        Parameters
        ----------
        params : dict
            Dictionary of configuration parameters.

        Returns
        -------
        scikit-learn classifier
            Ada Boost classifier defined in scikit-learn.
    """
    dtree_model = AdaBoostClassifier(
        n_estimators=params['n_estimators'], learning_rate=params['learning_rate'], algorithm=params['algorithm'],
        random_state=1
    )

    return dtree_model


def _classifier_knn(params):
    """ Instantiate and return K Nearest Neighbours classifier.

        Parameters
        ----------
        params : dict
            Dictionary of configuration parameters.

        Returns
        -------
        scikit-learn classifier
            K Nearest Neighbours classifier defined in scikit-learn.
    """
    knn_model = KNeighborsClassifier(
        n_neighbors=params['n_neighbors'], weights=params['weights'], p=params['p']
    )

    return knn_model


def _classifier_qda():
    """ Instantiate and return Quadratic Discriminant Analysis classifier.

        Parameters
        ----------
        params : dict
            Dictionary of configuration parameters.

        Returns
        -------
        scikit-learn classifier
            Quadratic Discriminant Analysis classifier defined in scikit-learn.
    """
    qda_model = QuadraticDiscriminantAnalysis()

    return qda_model


def _classifier_mlp(params):
    """ Instantiate and return Multi-layer Perceptron classifier.

        Parameters
        ----------
        params : dict
            Dictionary of configuration parameters.

        Returns
        -------
        scikit-learn classifier
            Multi-layer Perceptron classifier defined in scikit-learn.

        Raises
        ------
        RuntimeError
            If the solver specified in the configuration is unknown.
    """
    from sklearn.neural_network import MLPClassifier

    if params['solver'] == 'sgd':
        mlp_model = MLPClassifier(
            hidden_layer_sizes=params['hidden_layer_sizes'], activation=params['activation'],
            batch_size=params['batch_size'], learning_rate_init=params['learning_rate_init'], solver=params['solver'],
            learning_rate=params['learning_rate'], power_t=params['power_t'], momentum=params['momentum'],
            nesterovs_momentum=params['nesterovs_momentum'], random_state=1, max_iter=200, warm_start=False
        )
    elif params['solver'] == 'adam':
        mlp_model = MLPClassifier(
            hidden_layer_sizes=params['hidden_layer_sizes'], activation=params['activation'],
            batch_size=params['batch_size'], learning_rate_init=params['learning_rate_init'], solver=params['solver'],
            beta_1=params['beta_1'], beta_2=params['beta_2'], random_state=1, max_iter=200, warm_start=False
        )
    else:
        raise RuntimeError('Unknown solver {}'.format(params['solver']))

    return mlp_model
