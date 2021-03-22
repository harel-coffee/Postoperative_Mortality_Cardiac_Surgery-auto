from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from model.utils import oversample, impute, align_ok_output, df_to_mat, over_sample_ts_output, aki_to_cat_2, \
    normalize_column
from preprocessing.preprocessing_prepost import extract_features_ident, calculate_mort4

path = Path(__file__).parent.parent
min_max_scaler = MinMaxScaler(feature_range=(0, 1))


def prepare_data_total(ok, pca, dataset_type, output_type, mort, operation, mode, neuron, epochs, batch_size):
    print('[INFO] LSTM setup: dataset_type={}, output_type={}, mortality={},'
          'operation={}, mode={}, neurons={}, epochs={}, batch size={}'.format(dataset_type, output_type, mort,
                                                                               operation, mode, neuron, epochs,
                                                                               batch_size))

    output = get_y_aki(data=dataset_type) if output_type == 'AKI' \
        else get_y_mortality(data=dataset_type, mort=mort, operation=operation)
    output = output.dropna()

    print('[INFO] Output ({}) length={}'.format(output_type, len(output)))
    print('[INFO] Output value counts', output['Y'].value_counts())
    print('[INFO] OK shape={}, PCA shape={}'.format(ok.shape, pca.shape))

    if mode != 'intra-only':
        try:
            ok = pd.merge(ok, pca, on='Entry', how='left')
        except ValueError as v:
            print(v)
            print(ok.dtypes)
            print(pca.dtypes)

    ids_ok = ok["Entry"].unique()
    ids_output = output["Entry"].unique()
    ids = pd.DataFrame({"Entry": list(set(ids_ok) & set(ids_output))})

    output = pd.merge(ids, output, on="Entry")
    train, test = split_train_test_entries(output["Entry"], output["Y"])
    output = output.set_index("Entry")

    y_train, y_test = get_train_test(train, test, output)
    intra_train, intra_test = None, None

    return intra_test, intra_train, ok, output, test, train, y_test, y_train


def prepare_train_test(x_train, x_test, y_train, y_test):
    # align with output
    y_train_arr, x_train = y_train.align(x_train, join='left', axis=0)
    y_test_arr, x_test = y_test.align(x_test, join='left', axis=0)

    print('[INFO] Aligned shapes: (%s, %s), (%s, %s)' % (
        x_train.shape, y_train_arr.shape, x_test.shape, y_test_arr.shape))

    # to matrix
    x_train = x_train.values
    x_test = x_test.values
    y_train_arr = y_train_arr["Y"].values
    y_test_arr = y_test_arr["Y"].values
    print('[INFO] Matrix shapes x_train and x_test:', x_train.shape, x_test.shape)
    print('[INFO] Matrix shapes y_train and y_test:', y_train_arr.shape, y_test_arr.shape)

    # impute
    x_train = impute(x_train)
    x_test = impute(x_test)
    print('[INFO] Imputed shapes (x):', x_train.shape, x_test.shape)

    # oversample train set
    x_train, y_train_arr = oversample(x_train, y_train_arr)
    y_train = y_train_arr
    y_test = y_test_arr
    print('[INFO] Oversampled shapes (x,y):', x_train.shape, y_train_arr.shape)

    return x_train, x_test, y_train, y_test, y_test_arr


def extract_post(train, test, dataset_type):
    post_ok = pd.read_csv(path / 'data/post_{}_data.csv'.format(dataset_type), index_col=[0]).set_index('Entry')

    # Normalizing post-operative data
    var = list(post_ok)
    for v in var:
        post_ok[v] = normalize_column(post_ok[v])
    print('[INFO] Normalized post-operative data...')

    postop_train = post_ok[post_ok.index.isin(train)]
    postop_test = post_ok[post_ok.index.isin(test)]
    return postop_test, postop_train


def get_y_aki(data='total'):
    if data == 'random500':
        aki = pd.read_csv(path / 'data500/AKI.txt')
        li = [aki]
    else:
        surgeries = ['CORON', 'KLCOR', 'AOKLP', 'XKLEP', 'AORTA']
        dates = ['1994', '1999', '2009']
        li = []
        for surgery in surgeries:
            for date in dates:
                dataset_name = path / ('data/' + surgery + '_' + date + '_AKI.txt')
                dataset = pd.read_csv(dataset_name)
                dataset['Operation type'] = surgery
                dataset['Entry'] = dataset['Recn'].astype(str) + date + '_' + surgery
                li.append(dataset)

    aki = pd.concat(li, axis=0, ignore_index=True)
    aki = aki[aki['AKI'].notna()]

    aki["AKI"][aki["AKI"] < 1.0] = 0.0
    aki["AKI"][aki["AKI"] >= 1.0] = 1.0

    aki["Y"] = aki["AKI"]

    aki = aki[["Entry", "Y"]]

    print(aki.head(5))
    return aki


def get_y_mortality(data='total', mort='5y', operation='TOTAL'):
    print('Mortality:', mort)
    ident = extract_features_ident(data=data).reset_index()
    mortality = "Overleden_" + mort

    ident['Datum'] = pd.to_datetime(ident['Datum'], format='%Y-%m-%d', errors='coerce')
    ident = ident.dropna(subset=['Datum'])
    ident['ZOverleden'] = pd.to_datetime(ident['ZOverleden'], format='%Y-%m-%d')
    ident['Overleden_4d'] = ident.apply(lambda x: calculate_mort4(x), axis=1)
    ident = ident.loc[~ident['Overleden_4d'], :]
    ident["Y"] = ident[mortality]
    if operation != 'TOTAL':
        ident = ident[ident['Operation type'] == operation]
    ident = ident[["Entry", "Y"]]
    return ident


def get_proba(classifier, x_test):
    y_proba = classifier.predict_proba(x_test)
    return y_proba


def get_train_test(train_IDs, test_IDs, df):
    train = df[df.index.isin(train_IDs)]
    test = df[df.index.isin(test_IDs)]

    return train, test


def get_train_test_ok(train_IDs, test_IDs, ok):
    ok = ok.fillna(-1)

    x_train = ok[ok["Entry"].isin(train_IDs)]
    x_test = ok[ok["Entry"].isin(test_IDs)]

    return x_train, x_test


def get_train_test_output(ok, mode='total', output_type='mortality', oversample=True):
    output = get_y_aki(data=mode) if output_type == 'AKI' else get_y_mortality(data=mode)

    ok = pd.DataFrame.merge(ok, output, on="Entry")
    print(len(ok["Entry"].unique()))

    y = ok[["Entry", "Y"]].drop_duplicates()
    ok = ok.drop(columns=["Y"])
    ok = ok.fillna(-1)
    train, test = split_train_test_entries(y["Entry"], y["Y"])

    y = y.set_index("Entry")

    x_train, x_test = get_train_test_ok(train, test, ok)
    y_train, y_test = get_train_test(train, test, y)

    x_train, y_train = align_ok_output(x_train, y_train)
    x_test, y_test = align_ok_output(x_test, y_test)

    x_train = df_to_mat(x_train)
    x_test = df_to_mat(x_test)
    print('[INFO] Train and test shape: ', x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    if oversample is True:
        x_train, y_train = over_sample_ts_output(x_train, y_train)
    if output_type == 'AKI':
        y_train_classes, y_train, y_test_classes, y_test = aki_to_cat_2(y_train, y_test)
        print('[INFO] AKI shape:')
        print(y_train_classes.shape, y_test_classes.shape)
        print(y_train.shape, y_test.shape)
    else:
        y_test_classes = y_test
        y_train_classes = y_train

    return x_train, y_train, x_test, y_test, y_train_classes, y_test_classes, train, test


def split_train_test_entries(ids, output_classes):
    train_ids, test_ids, _, _ = train_test_split(ids, output_classes, test_size=0.2, random_state=1,
                                                 stratify=output_classes)
    return train_ids, test_ids
