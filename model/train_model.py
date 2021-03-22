"""
This file contains the src for the normal LSTM predicting mortality based on intra-operative data.
"""
from functools import reduce
from pathlib import Path

import pandas as pd
from tensorflow import set_random_seed

from model.data_preparation import get_train_test_output, extract_post, prepare_train_test, prepare_data_total
from model.lstm_classifiers import fit_lstm_aki_model, fit_lstm_model
from model.lstm_utils import lstm_classification_generate_results, get_full_representation
from model.other_classifiers import classify_xgboost
from model.utils import save_lstm_files, save_train_test, save_model_output

set_random_seed(1)
path = Path(__file__).parent.parent


# Predict based on intra-operative time series data in LSTM classifier
def predict_intra(ok, experiment_dir, dataset_type, exp_name, epochs=100, batch_size=264, neuron=512, dropout=0,
                  r_dropout=0, output_type='mortality'):
    oversample = False if output_type == 'AKI' else True
    x_train, y_train, x_test, y_test, y_train_classes, y_test_classes, train_id, test_id = \
        get_train_test_output(ok, mode=dataset_type, output_type=output_type, oversample=oversample)

    if output_type == 'AKI':
        model, history = fit_lstm_aki_model(x_train, y_train, x_test, y_test, epochs, neuron, batch_size,
                                            dropout, experiment_dir)
    else:
        model, history = fit_lstm_model(x_train, y_train_classes, x_test, y_test_classes, epochs, batch_size, neuron,
                                        dropout=dropout)

    save_lstm_files(experiment_dir, test_id, train_id)

    return lstm_classification_generate_results(model, history, exp_name, experiment_dir, x_test, y_test_classes,
                                                output_type, neuron, epochs, batch_size, dropout, r_dropout)


# Predict based on peri-operative data in LSTM for next input prediction and XGBoost classifier
def predict_total(ok, experiment_dir, dataset_type, pca, epochs, batch_size, neuron, dropout, r_dropout,
                  output_type='mortality', mode='all', mort='5y', operation=None, euroscore=False):
    intra_test, intra_train, ok, output, test, train, y_test, y_train = prepare_data_total(ok=ok, pca=pca,
                                                                                           dataset_type=dataset_type,
                                                                                           output_type=output_type,
                                                                                           mort=mort,
                                                                                           operation=operation,
                                                                                           mode=mode, neuron=neuron,
                                                                                           epochs=epochs,
                                                                                           batch_size=batch_size)
    if mode == 'all' or mode == 'intra' or mode == 'intra-only':
        intra_train, intra_test = get_full_representation(ok, output, train, test, epochs,
                                                          neuron, batch_size, dropout, r_dropout,
                                                          experiment_dir)

    return classify_total(intra_train, intra_test, y_train, y_test, train, test, experiment_dir, batch_size, neuron,
                          epochs, dropout, r_dropout, mode, dataset_type, euroscore)


# XGBoost classifier triggering method
def classify_total(intra_train, intra_test, y_train, y_test, train, test, experiment_dir, batch_size, neuron, epochs,
                   dropout, r_dropout, mode='all', dataset_type='total', euroscore=False):
    postop_test, postop_train = extract_post(train, test, dataset_type)
    if mode == 'all':
        x_train = reduce(lambda left, right: pd.merge(left, right, on="Entry"),
                         [intra_train, postop_train])
        x_test = reduce(lambda left, right: pd.merge(left, right, on="Entry"),
                        [intra_test, postop_test])
    elif mode == 'intra' or mode == 'intra-only':
        x_train = intra_train
        x_test = intra_test
    else:  # mode == post
        x_train = postop_train
        x_test = postop_test

    x_train, x_test, y_train_classes, y_test_classes, y_test_arr = prepare_train_test(x_train, x_test, y_train, y_test)

    # classify using xgboost
    results, model_xgb = classify_xgboost(x_train, y_train_classes, x_test, y_test_classes,
                                          [epochs, batch_size, dropout, r_dropout, neuron])
    print(results)

    save_train_test(x_train, x_test, y_train_classes, y_test_arr, train, test, experiment_dir)
    save_model_output(model_xgb, results, experiment_dir, euroscore)

    return results
