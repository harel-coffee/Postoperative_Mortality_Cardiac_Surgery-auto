import xgboost as xgb
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.utils import plot_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from analyze_model_performance.scores import score_list


def classify_logistic_regression(x_train, y_train, x_test, y_test, y_test_ohe, output='mort'):
    if output == 'AKI':
        clf = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')
    else:
        clf = LogisticRegression(random_state=0, class_weight='balanced', solver='sag')

    clf.fit(x_train, y_train)

    accuracy = clf.score(x_test, y_test)
    prediction = clf.predict(x_test)
    y_proba = clf.predict_proba(x_test)

    f1 = f1_score(y_test, prediction, average='macro')
    auc = roc_auc_score(y_test_ohe, y_proba, average='macro')

    print(classification_report(y_test, prediction))

    score = [accuracy, f1, auc]

    return score


def classify_mlp(x_train, y_train, x_test, y_test, output='mortality'):
    n_output = y_train.shape[1] if output == 'AKI' else 1
    loss_function = 'categorical_crossentropy' if output == 'AKI' else 'binary_crossentropy'
    model = Sequential()
    model.add(Dense(24, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(n_output, activation='sigmoid'))

    model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])
    plot_model(model, to_file='MLP_plot.png', show_shapes=True, show_layer_names=True)
    history = model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_test, y_test))

    score = score_list(model, None, x_test, y_test)

    return model, history, score


# gradient boosting
def classify_xgboost(x_train, y_train, x_test, y_test, params):
    cv_params = {
        'min_child_weight': [5],
        'max_depth': [7],
        'scale_pos_weight': [2.5],
        'max_delta_step': [1],
        'learning_rate': [0.01],
        'n_estimators': [300],
        'subsample': [0.8]
    }
    grid_search_xgb = GridSearchCV(xgb.XGBClassifier(random_state=0),
                                   cv_params, cv=10, n_jobs=-1, verbose=3, scoring='roc_auc')
    grid_search_xgb.fit(x_train, y_train)
    print(grid_search_xgb.best_estimator_)
    model = grid_search_xgb.best_estimator_
    model.fit(x_train, y_train)
    score, results = score_list(model, params, x_test, y_test)
    return results, model
