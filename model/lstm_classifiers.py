from random import seed

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import optimizers, models, layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import AUC
from tensorflow.keras.regularizers import L1L2

from analyze_model_performance.plots import plot_loss
from model.lstm_utils import fit_next_input_predictor, extract_encoder_output
from model.utils import split_last

seed(1)


def fit_lstm_model(train_x, train_y, test_x, test_y, n_epochs, batch_size, n_lstm, dropout=0, r_dropout=0, lrate=0.0001,
                   output='mort'):
    class_weight = compute_class_weight('balanced', np.unique(train_y), train_y)
    print(class_weight)

    n_output = train_y.shape[1] if output == 'AKI' else 1

    model = models.Sequential()
    model.add(layers.Masking(mask_value=-1, input_shape=(None, train_x.shape[2])))
    model.add(layers.LSTM(n_lstm, dropout=dropout, recurrent_dropout=r_dropout,
                          kernel_regularizer=L1L2(l1=0.0, l2=1e-6),
                          recurrent_regularizer=L1L2(l1=0.0, l2=1e-6)))

    model.add(layers.Dense(n_output, activation='sigmoid'))
    opt = optimizers.RMSprop(lr=lrate)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', AUC()])

    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    mc = ModelCheckpoint('../training_data/best_model.h5', monitor='val_loss', mode='min', verbose=1,
                         save_best_only=True)
    print(model.summary())

    history = model.fit(train_x, train_y, epochs=n_epochs, batch_size=batch_size, validation_data=(test_x, test_y))
    return model, history


def fit_lstm_aki_model(train_x, train_y, test_x, test_y, n_epochs, n_lstm, batch_size, dropout, experiment_dir):
    print('Train and test shape AKI', train_x.shape, train_y.shape)
    # class_weight = compute_sample_weight('balanced', train_y)
    # print(class_weight)
    # print('Class weight', class_weight)

    model = models.Sequential()
    model.add(layers.Masking(mask_value=-1, input_shape=(None, train_x.shape[2])))
    model.add(layers.Bidirectional(layers.LSTM(n_lstm, dropout=dropout, kernel_regularizer=L1L2(l1=0.0, l2=1e-6),
                                               recurrent_regularizer=L1L2(l1=0.0, l2=1e-6))))
    model.add(layers.Dense(train_y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    dir = experiment_dir + '/best_model.h5'
    mc = ModelCheckpoint(dir, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    history = model.fit(train_x, train_y, epochs=n_epochs, batch_size=batch_size, validation_data=(test_x, test_y),
                        callbacks=[es, mc])
    return model, history


def fit_lstm_next_step(train_x, test_x, epoch=100, batch_size=264, neuron=512, dropout=0, r_dropout=0,
                       experiment_dir='training_data'):
    train_x_sub, train_y_s = split_last(train_x)
    test_x_sub, test_y_s = split_last(test_x)

    model, history = fit_next_input_predictor(train_x_sub, train_y_s, test_x_sub, test_y_s, epoch, batch_size,
                                              neuron, dropout, experiment_dir)

    plot_loss(history=history, n_lstm=neuron, n_epochs=epoch, dropout=dropout, r_dropout=r_dropout, lrate=0.01,
              experiment_dir=experiment_dir)

    pred_train_x, model_train = extract_encoder_output(model, train_x, 1, experiment_dir)
    pred_test_x, model_test = extract_encoder_output(model, test_x, 1, experiment_dir)

    return train_x_sub, train_y_s, model, model_train, model_test, history, pred_train_x, pred_test_x
