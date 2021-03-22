import pandas as pd
from numpy.random import seed
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.experimental import CosineDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.callbacks import ModelCheckpoint

from analyze_model_performance.plots import plot_loss_acc
from analyze_model_performance.scores import score_list
from model import utils, data_preparation, lstm_classifiers

seed(1)


def fit_next_input_predictor(seq_in, out, seq_in_test, out_test, epochs=10, batch_size=64, neurons=256, dropout=0,
                             experiment_dir='training_data'):
    n_features = seq_in.shape[2]

    model = models.Sequential()

    model.add(layers.Masking(mask_value=-1, input_shape=(None, n_features)))
    model.add(layers.LSTM(neurons, dropout=dropout, kernel_regularizer=L1L2(l1=0.0, l2=1e-6),
                          recurrent_regularizer=L1L2(l1=0.0, l2=1e-6)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(n_features))
    plot_model(model, to_file=(experiment_dir + '/SR_LSTM_plot.png'), show_shapes=True, show_layer_names=True)
    lr_schedule = CosineDecay(initial_learning_rate=1e-3, decay_steps=epochs, alpha=1e-6)
    opt = Adam(learning_rate=lr_schedule, amsgrad=False)
    model.compile(loss='mae', optimizer=opt, metrics=['accuracy'])
    dir = experiment_dir + '/best_model.h5'
    mc = ModelCheckpoint(dir, monitor='val_loss', mode='min', verbose=1,
                         save_best_only=True)
    print(model.summary())
    history = model.fit(seq_in, out, epochs=epochs, batch_size=batch_size, validation_data=(seq_in_test, out_test),
                        callbacks=[mc])

    return model, history


def extract_encoder_output(model, input_seq, layer_nr, experiment_dir):
    # connect the encoder LSTM as the output layer
    model = models.Model(inputs=model.inputs, outputs=model.layers[layer_nr].output)
    plot_model(model, show_shapes=True, to_file=(experiment_dir + '/lstm_encoder.png'))

    # get the feature vector for the input sequence
    y_hat = model.predict(input_seq)

    return y_hat, model


def get_full_representation(ok, output, train_ids, test_ids, epoch, neuron, batch_size, dropout, r_dropout,
                            experiment_dir):
    x_train, x_test = data_preparation.get_train_test_ok(train_ids, test_ids, ok)
    y_train, y_test = data_preparation.get_train_test(train_ids, test_ids, output)

    x_train, y_train = utils.align_ok_output(x_train, y_train)
    x_test, y_test = utils.align_ok_output(x_test, y_test)
    print('Preparing train and test shape: ', x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    x_train = utils.df_to_mat(x_train)
    x_test = utils.df_to_mat(x_test)
    print('Train and test shape: ', x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    x_train_input, y_train_input, model, model_train, model_test, \
    history, pred_train, pred_test = lstm_classifiers.fit_lstm_next_step(x_train, x_test, epoch=epoch, neuron=neuron,
                                                                         batch_size=batch_size, dropout=dropout,
                                                                         r_dropout=r_dropout,
                                                                         experiment_dir=experiment_dir)

    train_rep = pd.DataFrame(pred_train, index=train_ids)
    train_rep.index.name = "Entry"

    test_rep = pd.DataFrame(pred_test, index=test_ids)
    test_rep.index.name = "Entry"

    utils.save_lstm_model_output(experiment_dir, model, model_test, model_train, pred_test, pred_train, test_rep,
                                 train_rep)

    return train_rep, test_rep


def lstm_classification_generate_results(model, history, exp_name, experiment_dir, x_test, y_test, output_type, neuron,
                                         epochs, batch_size, dropout, r_dropout):
    score, results = score_list(model, [epochs, batch_size, dropout, r_dropout, neuron], x_test, y_test,
                                output_type)
    plot_loss_acc(history, score[0], n_lstm=neuron, n_epochs=epochs, dropout=dropout, experiment_dir=experiment_dir)

    model_name = experiment_dir + '/' + exp_name + '.h5'
    model.save(model_name)
    return results
