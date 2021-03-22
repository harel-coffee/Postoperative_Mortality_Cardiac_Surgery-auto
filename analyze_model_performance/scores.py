from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score, confusion_matrix, \
    brier_score_loss
import numpy as np
import pandas as pd


def score_list(classifier, params, x_test, y_test, output_type='mortality'):
    if hasattr(classifier, 'predict_classes'):
        y_pred = classifier.predict_classes(x_test)
    else:
        y_pred = classifier.predict(x_test)
    y_proba = classifier.predict_proba(x_test)
    y_proba = y_proba if y_proba.shape[1] <= 1 else y_proba[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(classification_report(y_test, y_pred))

    sens = sensitivity(y_test, y_pred)
    spec = specificity(y_test, y_pred)
    j = youden_index(sens, spec)
    if output_type == 'mortality':
        auc = roc_auc_score(y_test, y_proba, average='macro')
        brier = brier_score_loss(y_test, y_pred)
        score = [accuracy, f1, auc, sens, spec, j, brier]
        columns = ["accuracy", "f1", "auroc",
                   "sens", "spec", "j", "brier",
                   "epochs", "batch size", "dropout", "r_dropout", "units"]
    else:
        score = [accuracy, f1, sens, spec, j]
        columns = ["accuracy", "f1", "sens", "spec", "j",
                   "epochs", "batch size", "dropout", "r_dropout", "units"]
    score = np.append(score, params)
    results = pd.DataFrame(data=[score], columns=columns)
    return score, results


def youden_index(sens, spec):
    return sens + spec - 1


def sensitivity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tpr = tp / (tp + fn)
    return tpr[1]


def specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    tnr = tn / (tn + fp)
    return tnr[1]


if __name__ == '__main__':
    y_pred = np.array([0.21, 0.32, 0.63, 0.35, 0.92, 0.79, 0.82, 0.99, 0.04])
    y_true = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0])
    sens = sensitivity(y_true, y_pred.round())
    spec = specificity(y_true, y_pred.round())
    print(sens)
    print(spec)
    print(youden_index(sens, spec))
