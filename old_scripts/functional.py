#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Helper functions. """

import matplotlib.pyplot as plt
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

CLASSIFIER_ORDER_ = ['mlp', 'svm_rbf', 'svm_polynomial', 'svm_linear', 'gp',
                     'random_forest', 'decision_tree', 'ada_boost', 'knn', 'qda']


def load_dataset_parameters(filename):
    """ Load dataset parameters in a .json file.

        Parameters
        ----------
        filename : str
            Absolute path to parameter file.

        Returns
        -------
        dict
            Dictionary of dataset parameters.
    """
    with open(filename) as json_file:
        d_params = json.load(json_file)

    return d_params


def compute_accuracy(model, X_train, y_train, X_test, y_test, class_labels):
    """ Fit a model to the training data and compute the accuracy of the predictions.

        Parameters
        ----------
        model : scikit-learn classifier
            Classification model.
        X_train : array_like
            Numpy array of training features.
        y_train : array_like
            Numpy array of training labels.
        X_test : array_like
            Numpy array of test features.
        y_test : array_like
            Numpy array of test labels.
        class_labels : array_like
            Array of the labels for each class.

        Returns
        -------
        tuple
            Predictions and their probabilities as well as the computed accuracy and F1 scores.
    """

    # Fit the data
    model.fit(X_train, y_train)

    # If predicting probabilities or applying decision function (svm)
    if hasattr(model, 'decision_function'):
        model_or = OneVsRestClassifier(model)
        y_train_bin = label_binarize(y_train, classes=class_labels)
        model_or.fit(X_train, y_train_bin)
        y_score = model_or.decision_function(X_test)
    else:
        y_score = model.predict_proba(X_test)

    # Get the predictions, accuracy score and f1 score
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')

    return pred, y_score, acc, f1


def compute_roc(model, y_test, y_score, class_labels):
    """ Compute the area under the Receiver Operating Characteristic curve (ROC AUC).

        Parameters
        ----------
        model : scikit-learn classifier
            Classification model.
        y_test : array_like
            Numpy array of test labels.
        y_score : array_like
            Numpy array of the probability of each class for each test sample.
        class_labels : array_like
            Array of the labels for each class.

        Returns
        -------
        tuple
            True and false positive rates as well as the ROC AUC.
    """

    # Compute micro-average ROC curve and ROC area
    y_test_bin = label_binarize(y_test, classes=class_labels)
    n_classes = y_test_bin.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # print(n_classes, y_test_bin.shape, y_score.shape)
    if y_score.ndim != y_test_bin.ndim:
        y_score = np.reshape(y_score, y_test_bin.shape)
    for i in range(n_classes):
        try:
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        except IndexError as ie:
            print(ie)
            raise ie
        except ValueError as ve:
            print(ve)
            raise ve
        roc_auc[i] = auc(fpr[i], tpr[i])

    # If model is a binary model
    if len(class_labels) == 2 and not hasattr(model, 'decision_function'):
        y_score = y_score[:, 1]

    # Get fpr, tpr and roc auc
    fpr['micro'], tpr['micro'], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    return tpr, fpr, roc_auc


def plot_roc(results, dataset_params):
    """ Plot Receiver Operating Characteristic curves.

        Two plots are generated. The first is the overall ROC curve for each classification model. The second is the
        ROC curve for each class in the dataset, shown separately for each classification model.

        Parameters
        ----------
        results : array_like
            Array storing all classification results.
        dataset_params : dict
            Dictionary of the parameters for the evaluated dataset.
    """

    # Set up the figures
    _, ax_all = plt.subplots()
    fig, ax = plt.subplots(nrows=2, ncols=5)
    ax = ax.flatten()

    # Combine the curves for each result on one plot
    for c in CLASSIFIER_ORDER_:
        for res in results:
            if res[0][0] == c:
                ax_all.plot(res[4]['micro'], res[3]['micro'],
                            lw=2, label='{} (area = {:.2f})'.format(res[0][1], res[5]['micro']))
    ax_all.set_xlim([0.0, 1.0])
    ax_all.set_ylim([0.0, 1.05])
    ax_all.set_xlabel('FP Rate')
    ax_all.set_ylabel('TP Rate')
    ax_all.set_title('ROC')
    ax_all.legend(loc='lower right')

    # Plot the curve for each different result
    legend_is_set = False
    for i in range(len(CLASSIFIER_ORDER_)):
        res = None
        for p in results:
            if p[0][0] == CLASSIFIER_ORDER_[i]:
                res = p
        if res is not None:
            if len(dataset_params['classes']) == 2:  # Binary classification
                ax[i+0].plot(res[4][0], res[3][0], color='black', linestyle=dataset_params['styles'][0], lw=1)
            else:
                for c in dataset_params['classes']:
                    ax[i+0].plot(res[4][c], res[3][c], color='black', linestyle=dataset_params['styles'][c],
                                 lw=1, label='{}'.format(dataset_params['labels'][c]))
                if not legend_is_set:
                    ax[i+0].legend(loc='lower right')
                    legend_is_set = True
        ax[i+0].set_xlim([0.0, 1.0])
        ax[i+0].set_ylim([0.0, 1.05])
        ax[i+0].set_xlabel('FP Rate')
        ax[i+0].set_ylabel('TP Rate')
        ax[i+0].set_aspect(1)
        if res is not None:
            ax[i+0].set_title('{}'.format(res[0][1]))
        else:
            ax[i+0].set_title('{}'.format(CLASSIFIER_ORDER_[i]))

    plt.subplots_adjust(wspace=0.3, hspace=0.0)
    plt.show()


def print_scores(label, y_test, pred, acc, f1, roc_auc):
    """ Print classification results to the terminal.

        Parameters
        ----------
        label : str
            Name of the classifier
        y_test : array_like
            Ground truth class labels.
        pred : array_like
            Predicted class labels.
        acc : float
            Accuracy value.
        f1 : float
            F1 score
        roc_auc : float
            Area under the Receiver Operating Characteristic curve.
    """
    print('Accuracy ({}): {:.2f}'.format(label, acc * 100))
    print('F1 ({}): {:.2f}'.format(label, f1 * 100))
    print(classification_report(y_test, pred))
    print(confusion_matrix(y_test, pred))
    print('ROC AUC')
    print(roc_auc)


def print_summary(results):
    """ Print final classification results to the terminal.

        Parameters
        ----------
        results : array_like
            Array storing all classification results.
    """
    print('----------')
    for c in CLASSIFIER_ORDER_:
        for p in results:
            if p[0][0] == c:
                print('{:20} : Acc - {:.2f}, F1 - {:.2f}, ROC AUC - {:.2f}'.format(
                    p[0][1], p[1] * 100, p[2] * 100, p[5]['micro'] * 100))
