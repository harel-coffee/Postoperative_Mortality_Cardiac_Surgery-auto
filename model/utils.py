"""
Some utility functions
"""
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from keras.utils import np_utils
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

path = Path(__file__).parent.parent
min_max_scaler = MinMaxScaler(feature_range=(0, 1))


# -------  Data preparation utils -------
def aki_to_cat(aki_list):
    encoder = LabelEncoder()
    encoder.fit(aki_list)
    encoded_y = encoder.transform(aki_list)
    # convert integers to dummy variables (i.e. one hot encoded)
    y = np_utils.to_categorical(encoded_y)
    return encoded_y, y


def aki_to_cat_2(train_y, test_y):
    encoder = LabelEncoder()
    encoder.fit(train_y)
    y_train_classes = encoder.transform(train_y)
    y_test_classes = encoder.transform(test_y)
    # convert integers to dummy variables (i.e. one hot encoded)
    y_train_ohe = np_utils.to_categorical(y_train_classes)
    y_test_ohe = np_utils.to_categorical(y_test_classes)
    return y_train_classes, y_train_ohe, y_test_classes, y_test_ohe


def normalize_column(series, scaler=min_max_scaler):
    series = series.values.reshape(-1, 1)
    s_scaled = scaler.fit_transform(series)
    s_scaled = s_scaled.reshape((-1))
    return s_scaled


def over_sample_ts_output(train_x, train_y):
    ids = np.arange(train_x.shape[0])
    print("IDs: ", ids)
    print(train_x.shape)
    ids = ids.reshape(-1, 1)

    ros = RandomOverSampler(random_state=1)
    train_ids, train_y = ros.fit_resample(ids, train_y)
    print(len(train_ids))
    train_set = np.empty((train_ids.shape[0], train_x.shape[1], train_x.shape[2]))

    for idx, Id in enumerate(train_ids):
        train_set[idx] = train_x[Id]  # np.append(train_set, trainX[Id], axis=0)

    print(train_set.shape)
    return train_set, train_y


def align_ok_output(x, y):
    x = pd.DataFrame.merge(x, y, on="Entry")

    y = x[["Entry", "Y"]].drop_duplicates()
    y = y["Y"].values
    x = x.drop(columns=["Y"])

    return x, y


def split_sequences(x_train, y_train_classes, training_length):
    sequences = []
    classes = []
    for (seq, mort) in zip(x_train, y_train_classes):
        seq = seq[~np.all(seq == -1, axis=1)]
        for i in range(training_length, len(seq)):
            extract = seq[i - training_length:i + 1]
            sequences.append(extract)
            classes.append(mort)

    sequences = np.array(sequences)
    classes = np.array(classes)

    return sequences, classes


def split_last(x_train):
    features = np.empty((0, x_train.shape[1] - 1, x_train.shape[2]))
    labels = []
    for seq in x_train:
        seq_no_nan = seq[~np.all(seq == -1, axis=1)]
        idx_last = seq_no_nan.shape[0] - 1
        seq = np.delete(seq, idx_last, 0)
        features = np.append(features, [seq], axis=0)
        labels.append(seq_no_nan[-1, :])

    features = np.asarray(features)
    labels = np.asarray(labels)
    print("split features", features.shape)
    print("split labels", labels.shape)

    return features, labels


def df_to_mat(df):
    g = df.groupby('Entry').cumcount()
    df = (df.set_index(['Entry', g])
          .unstack(fill_value=-1)
          .stack().groupby(level=0)
          .apply(lambda x: x.values.tolist())
          .tolist())
    df = np.asarray(df)
    return df


def impute(mat):
    arr = mat.tolist()
    imp = IterativeImputer(max_iter=10, random_state=0, skip_complete=True)
    mat = imp.fit_transform(arr)
    return np.array(mat)


def oversample(x, y):
    ros = RandomOverSampler(random_state=1)
    x, y = ros.fit_resample(x, y)

    return x, y


# ------- Saving files -------

def save_model_output(model_xgb, results, experiment_dir, euroscore):
    name = 'eur' if euroscore else 'default'
    # save model and results
    joblib.dump(model_xgb, (experiment_dir + '/model_xgb_{}.joblib'.format(name)))
    with open(experiment_dir + "/results_tot_{}.pkl".format(name), 'wb') as res:
        pickle.dump(results, res)
    results.to_csv(experiment_dir + '/results_{}.csv'.format(name))


def save_lstm_model_output(experiment_dir, model, model_test, model_train, pred_test, pred_train, test_rep, train_rep):
    # save output lstm
    np.save(experiment_dir + '/pred_train.npy', pred_train)
    np.save(experiment_dir + '/pred_test.npy', pred_test)

    # save pandas output
    train_rep.to_csv(experiment_dir + '/pred_train.csv')
    test_rep.to_csv(experiment_dir + '/pred_test.csv')

    # save models
    model_name = experiment_dir + '/main_model.h5'
    model.save(model_name)

    model_train_name = experiment_dir + '/train_model.h5'
    model_train.save(model_train_name)

    model_test_name = experiment_dir + '/test_model.h5'
    model_test.save(model_test_name)


def save_train_test(x_train, x_test, y_train_classes, y_test_arr, train, test, experiment_dir):
    # save train and test
    np.save(experiment_dir + '/train.npy', train)
    np.save(experiment_dir + '/test.npy', test)
    np.save(experiment_dir + '/x_train_gb.npy', x_train)
    np.save(experiment_dir + '/x_test_gb.npy', x_test)
    np.save(experiment_dir + '/y_train_gb.npy', y_train_classes)
    np.save(experiment_dir + '/y_test_gb.npy', y_test_arr)


def save_lstm_files(experiment_dir, test, train):
    np.save(experiment_dir + '/train_id.npy', train)
    np.save(experiment_dir + '/test_id.npy', test)
