#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Script for training and evaluating the performance of classification models. """

import argparse
import pickle
import functional as F

if __name__ == '__main__':
    # === Parse arguments
    parser = argparse.ArgumentParser(description='Plot ROC curves for stored data file')
    parser.add_argument('--file', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    results_processed = pickle.load(open(args.file, "rb"))
    F.print_summary(results_processed)

    dataset_params = F.load_dataset_parameters('dataset_parameters.json')
    dataset_params = dataset_params[args.dataset]
    F.plot_roc(results_processed, dataset_params)
