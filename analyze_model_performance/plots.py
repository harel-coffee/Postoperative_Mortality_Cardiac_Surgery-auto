from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
from pathlib import Path

path = Path(__file__).parent.parent


# -------- Plots  ---------
def plot_loss(history, n_lstm, n_epochs, dropout, r_dropout, lrate, experiment_dir):
    pyplot.rcParams.update({'font.size': 16})
    textstr = 'LSTM units: %s\nEpochs: %s\nLearning: %s\nDropout: %s\nRec. dropout: %s\n' % (
        n_lstm, n_epochs, lrate, dropout, r_dropout)
    _, ax = pyplot.subplots()

    # plot loss
    ax.plot(history.history['loss'])
    # ax.plot(history.history['val_loss'])
    ax.set_title('model train vs validation loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    # ax.legend(['train', 'validation'], loc='upper right')
    ax.text(0.05, 0.5, textstr, fontsize=12, transform=pyplot.gcf().transFigure)

    pyplot.subplots_adjust(left=0.30)
    pyplot.savefig(path / (experiment_dir + "/loss.png"))
    pyplot.close()


def plot_loss_acc(history, acc, n_lstm, n_epochs, experiment_dir, name='hemo', dropout=0, r_dropout=0, lrate=0.001):
    pyplot.rcParams.update({'font.size': 16})
    textstr = 'LSTM units: %s\nEpochs: %s\nLearning: %s\nDropout: %s\nRec. dropout: %s\nAccuracy: %.2f\n' % (
        n_lstm, n_epochs, lrate, dropout, r_dropout, acc)
    _, ax = pyplot.subplots(1, 2, figsize=(30, 15))

    # plot loss
    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])
    ax[0].set_title('model train vs validation loss')
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['train', 'validation'], loc='upper right')
    ax[0].text(0.05, 0.5, textstr, fontsize=16, transform=pyplot.gcf().transFigure)

    # plot acc
    print(history.history)
    ax[1].plot(history.history['acc'])
    ax[1].plot(history.history['val_acc'])
    ax[1].set_title('model train vs validation accuracy')
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['train', 'validation'], loc='upper right')

    pyplot.subplots_adjust(left=0.20)
    pyplot.savefig(path / (experiment_dir + "/loss_acc.png"))
    pyplot.close()


def plot_acc(history):
    pyplot.plot(history.history['acc'])
    pyplot.plot(history.history['val_acc'])
    pyplot.title('model train vs validation accuracy')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap='Blues',
                          count=0):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data_
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = pyplot.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    pyplot.savefig("../plots/conf_LSTM_AE" + str(count))
    pyplot.close()
    return ax
