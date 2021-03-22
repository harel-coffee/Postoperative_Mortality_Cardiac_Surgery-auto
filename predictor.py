import argparse
import os
import sys
import warnings

import pandas as pd

from model.train_model import predict_intra, predict_total
from preprocessing.preprocessing_intra import filtering
from preprocessing.preprocessing_prepost import preprocessing

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)


def train_model(exp_name, epochs, neurons, batch_size, dropout, mode='intra-only', mort='5y', operation='TOTAL',
                euroscore=False):
    if mode is None:
        modes = ['post', 'intra', 'intra-only', 'all']
    else:
        modes = [mode]
    if mort is None:
        mortalities = ['30d', '1y', '5y']
    else:
        mortalities = [mort]
    if operation is None:
        operations = ['TOTAL', 'CORON', 'KLEP', 'KLCOR']
    else:
        operations = [operation]

    for mort in mortalities:
        for mode in modes:
            for operation in operations:
                experiment_dir = 'training_data/{}_{}_{}_{}_{}'.format(exp_name, output_type, mode, str(operation),
                                                                       mort)
                if not os.path.isdir(experiment_dir):
                    os.makedirs(experiment_dir)
                if model_type == 'intra':
                    results = predict_intra(ok=ok, experiment_dir=experiment_dir,
                                            dataset_type=dataset_type, exp_name=exp_name, epochs=epochs,
                                            batch_size=batch_size, neuron=neurons, r_dropout=r_dropout,
                                            output_type=output_type)
                elif model_type == 'total':
                    results = predict_total(ok=ok, experiment_dir=experiment_dir,
                                            dataset_type=dataset_type, pca=pca,
                                            epochs=epochs, neuron=neurons,
                                            batch_size=batch_size, dropout=dropout, r_dropout=r_dropout,
                                            output_type=output_type, mode=mode, operation=operation,
                                            mort=mort, euroscore=euroscore)
                else:
                    print('No model defined!')
                    sys.exit()
                print(results)
                results.to_csv(experiment_dir + '/results.csv')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training model")
    parser.add_argument('--dataset_type', default='random500',
                        help='random500 or total')
    parser.add_argument('--model_type', default='total',
                        help='intra (LSTM classifier only), total (LSTM for next input prediction + XGBoost classifier')
    parser.add_argument('--exp_name', default=None, help='Experiment name')
    parser.add_argument('--epochs', default=1, type=int, help='epochs')
    parser.add_argument('--neurons', default=16, type=int, help='neurons')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--dropout', default=0, type=float, help='dropout')
    parser.add_argument('--r_dropout', default=0, type=float, help='dropout')
    parser.add_argument('--output_type', default='mortality', help='AKI or mortality')
    parser.add_argument('--mode', default='all', help='all, intra-only, intra or post')
    parser.add_argument('--mortality', default='30d', help='30d (30-day), 1y (1-year) or 5y (5-year)')
    parser.add_argument('--operation', default='TOTAL', help='Type of operation: CORON (CABG), KLEP (valve), '
                                                             'KLCOR (combined) or TOTAL (all)')
    parser.add_argument('--euroscore', default=False, help='Include EuroSCORE')

    args = parser.parse_args()

    # assign arguments to global variables
    dataset_type = args.dataset_type
    model_type = args.model_type
    exp_name = args.exp_name
    epochs = args.epochs
    neurons = args.neurons
    batch_size = args.batch_size
    dropout = args.dropout
    r_dropout = args.r_dropout
    output_type = args.output_type
    mode = args.mode
    mortality = args.mortality
    operation = args.operation
    euroscore = args.euroscore

    try:
        ok = pd.read_csv('data/' + dataset_type + '.csv', index_col=[0])
        pca = pd.read_csv('data/' + dataset_type + '_pca.csv', index_col=[0])
    except OSError as e:
        print('No preprocessed data files found. Running preprocessing...')
        ok = filtering(data=dataset_type, features='total', name=dataset_type)
        preprocessing('pre', data=dataset_type)
        preprocessing('post', data=dataset_type, euroscore=euroscore)
        pca = pd.read_csv('data/' + dataset_type + '_pca.csv', index_col=[0])

    if exp_name is None:
        exp_name = '{}_{}_{}_{}_{}_{}'.format(dataset_type, model_type, epochs, neurons, batch_size, dropout)
    train_model(exp_name, epochs, neurons, batch_size, dropout, mode, mortality, operation, euroscore)
