import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler

from preprocessing import var_values

pd.options.mode.chained_assignment = None  # default='warn'
path = Path(__file__).parent.parent


def calculate_mort(row):
    if (row['ZOverleden'] is np.nan) or (row['Datum'] is np.nan):
        return np.nan
    elif (row['ZOverleden'] - row['Datum']).days <= 30:
        return '30'
    elif (row['ZOverleden'] - row['Datum']).days / 365.25 <= 1:
        return '1'
    elif (row['ZOverleden'] - row['Datum']).days / 365.25 > 1:
        return '5'
    else:
        return np.nan


def calculate_mort4(row):
    if (row['ZOverleden'] - row['Datum']).days <= 4:
        return True
    else:
        return False


def calculate_mort30(row):
    if (row['ZOverleden'] - row['Datum']).days <= 30:
        return True
    else:
        return False


def calculate_mort1(row):
    if (row['ZOverleden'] - row['Datum']).days / 365.25 <= 1:
        return True
    else:
        return False


def calculate_mort4(row):
    if (row['ZOverleden'] - row['Datum']).days <= 4:
        return True
    else:
        return False


def calculate_mort5(row):
    if (row['ZOverleden'] - row['Datum']).days / 365.25 <= 5:
        return True
    else:
        return False


def calculate_period(row):
    if pd.isnull(row['ZOverleden']):
        return (datetime(2017, 12, 1) - row['Datum']).days
    else:
        return (row['ZOverleden'] - row['Datum']).days


def read_ident(data='total', from_saved=True):
    """
    Read demographic data
    @return: dataframe
    """
    if data == 'random500':
        ident = pd.read_csv(path / 'data500/ident.txt')
        li = [ident]
        ident = pd.concat(li, axis=0, ignore_index=True)
        ident = preprocess_ident_df(ident)
    elif from_saved is True:
        ident = pd.read_csv(path / 'data/complete_ident.csv', index_col=0)
        # ident = ident[~ident.index.duplicated()]
        print('length of IDENT', len(ident))
        print(ident['Operation type'].value_counts())
        print('FROM SAVED FILE')
    else:
        surgeries = ['CORON', 'KLCOR', 'AOKLP', 'XKLEP', 'AORTA']
        dates = ['1994', '1999', '2009']
        li = []
        for surgery in surgeries:
            for date in dates:
                dataset_name = '../data/' + surgery + '_' + date + '_ident.txt'
                dataset = pd.read_csv(dataset_name)
                dataset['Operation type'] = surgery
                dataset['Entry'] = dataset['Entry'].astype(str) + date + '_' + surgery
                li.append(dataset)

        ident = pd.concat(li, axis=0, ignore_index=True)
        ident = preprocess_ident_df(ident)
    return ident


def preprocess_ident_df(ident):
    print('len', len(ident))
    ident['Overleden'] = ident['ZOverleden'].notna()
    ident['Datum'] = pd.to_datetime(ident['Datum'], format='%d%m%y', errors='coerce')

    ident = ident.dropna(subset=['Datum'])

    ident['ZOverleden'] = pd.to_datetime(ident['ZOverleden'], format='%d-%m-%Y')
    ident['Overleden_4d'] = ident.apply(lambda x: calculate_mort4(x), axis=1)
    ident['Overleden_30d'] = ident.apply(lambda x: calculate_mort30(x), axis=1)
    ident['Overleden_1y'] = ident.apply(lambda x: calculate_mort1(x), axis=1)
    ident['Overleden_5y'] = ident.apply(lambda x: calculate_mort5(x), axis=1)
    ident['Follow up'] = ident.apply(lambda x: calculate_period(x), axis=1)
    return ident


def extract_features_ident(data='total', ident=None, from_saved=True):
    """
    Extract features from demographic dataset
    @return: extracted features
    """
    if ident is None:
        ident = read_ident(data, from_saved)
    print('[INFO] In ident total: ', len(ident))
    # convert to binary
    ident['Geslacht'] = ident['Geslacht'].replace({'M': 0, 'Man': 0, 'V': 1, 'Vrouw': 1})
    ident = ident.reindex()

    # delete invalid operation times
    ident.loc[ident['Tijd Oper'] > 1440, 'Tijd Oper'] = np.nan
    ident.loc[ident['Tijd Perf'] > 1440, 'Tijd Perf'] = np.nan
    ident.loc[ident['Tijd Aocc'] > 1440, 'Tijd Aocc'] = np.nan

    if data != 'random500':
        # Delete age '-'
        ident['Leeftijd'] = ident['Leeftijd'].replace({'-': np.nan})

    ident['Leeftijd'] = ident['Leeftijd'].astype('float').abs()

    # Calculate BMI
    bmi = (ident['Gewicht'] / (ident['Oppervlakte'] ** 2)).round(2)
    ident.insert(6, 'BMI', bmi, True)
    # ident.to_csv('../data/complete_ident.csv')

    # calculate eCCR for EuroSCORE II
    df_prepost = read_prepost(data)[['Entry', 'Status', 'ZLKreat']]
    df_prepost = df_prepost[df_prepost['Status'] == 'LAB <']
    df_prepost = df_prepost.groupby('Entry').agg(lambda x: get_mean(x))

    ident = pd.merge(ident, df_prepost, on='Entry', how='left')
    sex = np.where((ident['Geslacht'] == 1.0), 0.85, 1.0)
    ident['eCCR'] = sex * ((140 - ident['Leeftijd']) / ident['ZLKreat']) * (ident['Gewicht'] / 72)
    ident = ident.drop(columns=['ZLKreat', 'Status'])
    ident = ident.set_index('Entry')
    return ident


def read_prepost(data='total'):
    """
    Read pre- and post-operative data
    @return: dataframe
    """
    if data == 'random500':
        prepost = pd.read_csv(path / 'data500/prepost.txt')
        li = [prepost]
        print('Read random500 prepost...')
    else:
        surgeries = ['CORON', 'KLCOR', 'AOKLP', 'XKLEP']
        dates = ['1994', '1999', '2009']
        li = []
        for surgery in surgeries:
            for date in dates:
                dataset_name = path / ('data/' + surgery + '_' + date + '_prepost.txt')
                dataset = pd.read_csv(dataset_name)
                dataset['Entry'] = dataset['Entry'].astype(str) + date + '_' + surgery
                li.append(dataset)
        print('Read all prepost...')
    prepost = pd.concat(li, axis=0, ignore_index=True)
    return prepost


def get_last(group):
    last_index = group.last_valid_index()
    return group[last_index] if last_index else -1


def get_percentual_change(current, previous):
    try:
        return (abs(current - previous) / previous) * 100.0
    except ZeroDivisionError:
        return float('inf')


def get_start_val(x, col):
    idx = x.first_valid_index()
    if idx is None:
        return np.nan
    print(col.iloc[idx])
    return col.iloc[idx]


def extract_features_extra(data='total'):
    prepost = read_prepost(data)
    intra = pd.read_csv(path / ('data/' + data + '-extra.csv'), index_col=0).groupby('Entry')
    ident = extract_features_ident(data, from_saved=True)
    prepost = calculate_creat_clearance(prepost, ident)

    pre = prepost[prepost['Status'] == 'LAB <']
    post = prepost[prepost['Status'] == 'LAB >']

    df = pd.DataFrame()
    post_features = pd.DataFrame()
    # eCCR
    eccr_pre = pre.groupby('Entry')['eCCR'].agg(lambda x: get_mean(x))
    eccr_post = post.groupby('Entry')['eCCR'].agg(lambda x: get_mean(x))
    df['Per-operative eCCR decrease'] = eccr_post - eccr_pre
    df['Per-operative eCCR ratio'] = eccr_post / eccr_pre
    post_features['Maximum post-operative creatinine'] = post.groupby('Entry')['ZLKreat'].max()

    # Creatinine
    creat_pre = pre.groupby('Entry')['ZLKreat'].agg(lambda x: get_mean(x))
    creat_post = post.groupby('Entry')['ZLKreat'].agg(lambda x: get_mean(x))
    df['Absolute difference in creatinine'] = abs(creat_post - creat_pre)
    df['Relative difference in creatinine'] = (creat_post - creat_pre) * 100 / creat_post
    df['Percentual difference in creatinine'] = get_percentual_change(creat_post, creat_pre)

    # LDH, Hb, Glucose
    post_features['Maximum post-operative LDH'] = post.groupby('Entry')['ZLDH'].max()
    post_features['Maximum post-operative Glucose'] = post.groupby('Entry')['ZLGlucose'].max()
    post_features['Minimum post-operative Hb'] = post.groupby('Entry')['ZLHb'].min()

    print(post_features)
    df = pd.merge(df, post_features, on='Entry', how='left')

    # # Intra
    intra_df = pd.DataFrame()
    intra_df['Entry'] = pd.Series(intra['Entry'].unique())
    for name, group in intra:
        try:
            start_op = int(group['start_operatie'].mean())
            end_op = int(group['end_operatie'].mean())
            if end_op < start_op:
                temp = start_op
                start_op = end_op
                end_op = temp
        except:
            continue
        start = int(group['start_perfusie'].mean())
        end = int(group['end_perfusie'].mean())
        if end < start:
            temp = start
            start = end
            end = temp
        try:
            index = pd.IntervalIndex.from_tuples([(start, end)], closed='neither')
            index_op = pd.IntervalIndex.from_tuples([(start_op, end_op)], closed='neither')
        except:
            print(start, end)
            continue
        vars_start_surgery = [('SBP at start surgery', 'PArtS'), ('DBP at start surgery', 'PArtD'),
                              ('PCVD at start surgery', 'PCVD'), ('Pulsox at start surgery', 'Pulsox'),
                              ('CETCO2 at start surgery', 'CETCO2'), ('HR at start surgery', 'HR_pulsOx')]
        vars_during_perf = [('SBP during perfusion', 'PArtS'), ('DBP during perfusion', 'PArtD'),
                            ('PCVD during perfusion', 'PCVD'), ('Pulsox during perfusion', 'Pulsox'),
                            ('CETCO2 during perfusion', 'CETCO2'), ('HR during perfusion', 'HR_pulsOx')]
        for k, v in vars_start_surgery:
            intra_df.at[name, k] = \
                group[v].groupby(pd.cut(group['Reltime'], bins=index_op)).first().iloc[0]
        for k, v in vars_during_perf:
            intra_df.at[name, k] = group[v].groupby(pd.cut(group['Reltime'], bins=index)).mean()
        intra_df.at[name, 'Duration of perfusion'] = end - start
        intra_df.at[name, 'Maximum CPB flow'] = group['F'].max()
        intra_df.at[name, 'Minimum body temperature'] = group['Thuid'].min()
    print(df)
    print(intra_df)
    intra_df = intra_df.drop(columns=['Entry'])
    df = pd.merge(df, intra_df, on='Entry', how='left')
    df.to_csv('../data/extra.csv')
    return df


def extract_features_pre(data='total'):
    prepost = read_prepost(data)
    prepost = prepost[prepost['Status'] == 'LAB <']
    prepost = prepost.filter(items=['Entry', 'ZLKreat', 'ZLUreum', 'ZLDH', 'ZLHb', 'ZLLeuco', 'ZLTrombo', 'ZLCa',
                                    'ZLALAT', 'ZLASAT', 'ZLBSE', 'sample_class'])
    prepost = prepost.replace({-1.0, np.nan})

    return prepost


def extract_features_post(data='total'):
    """
    extracting features from post-operative data of CORON datasets
    @return: resulting dataframe
    """
    prepost = read_prepost(data)
    # get last value of each variable after OK
    prepost = prepost[prepost['Status'] == 'LAB >']
    prepost = prepost.drop(columns=['Utime', 'Status', 'ZLAfdeling', 'ZLAfn_dt', 'ZLGlucKw', 'ZLAfn_tm', 'ZLkind'])
    prepost = prepost.replace({-1.0: np.nan})
    prepost = prepost.filter(items=['Entry', 'ZLLeuco', 'ZLKreat', 'ZLDH', 'ZLTrombo', 'ZLALAT', 'ZLASAT', 'ZLUreum',
                                    'ZLHb', 'ZLMononucE', 'ZLLymfoE', 'ZLNeutrE', 'ZLGlucose', 'sample_class'])
    return prepost


def extract_features(mode, data):
    """
    Extract features
    @param data: total or random500
    @param mode: pre or post
    @return: dataframe with features
    """
    if mode == 'pre':
        df = extract_features_pre(data)
    elif mode == 'post':
        df = extract_features_post(data)
    elif mode == 'extra':
        df = extract_features_extra(data)
    return df


def get_mean(group):
    if np.issubdtype(group.dtype, np.number):
        return np.nanmean(group)

    last_index = group.last_valid_index()
    return group[last_index] if last_index else -1


def calculate_creat_clearance(prepost, ident):
    """
    Calculate eCCR
    @param prepost: dataframe
    @param ident: demographic data for height, etc.
    @return: dataframe with eCCR column
    """
    df = pd.merge(prepost, ident, on='Entry')
    sex = np.where((df['Geslacht'] == 1.0), 0.85, 1.0)
    prepost['eCCR'] = sex * ((140 - df['Leeftijd']) / df['ZLKreat']) * (df['Gewicht'] / 72)
    return prepost


def calculate_periods(prepost, mode='pre'):
    """
    Calculate the time frames of medical measurements
    @param prepost: dataframe
    @param mode: pre or post
    @return: dataframe with new faetures
    """
    if mode == 'pre':
        features = ['ZLKreat', 'ZLUreum', 'ZLDH', 'ZLHb', 'ZLLeuco', 'ZLTrombo', 'ZLCa', 'ZLALAT', 'ZLASAT', 'ZLBSE',
                    'eCCR']
        for f in features:
            wtn24 = f + '_wtn24'
            # within 24 hours before surgery
            prepost[wtn24] = np.where((prepost['sample_class'] == 'L_pre24'), prepost[f], np.nan)
    elif mode == 'post':
        features = ['ZLLeuco', 'ZLKreat', 'ZLDH', 'ZLTrombo', 'ZLALAT', 'ZLASAT', 'ZLUreum', 'ZLHb', 'ZLMononucE',
                    'ZLLymfoE', 'ZLNeutrE', 'ZLGlucose', 'eCCR']
        for f in features:
            prepost[f + '_pos06'] = np.where((prepost['sample_class'] == 'L_pos06'), prepost[f], np.nan)
            prepost[f + '_pos12'] = np.where((prepost['sample_class'] == 'L_pos12'), prepost[f], np.nan)
            prepost[f + '_pos24'] = np.where((prepost['sample_class'] == 'L_pos24'), prepost[f], np.nan)
            prepost[f + '_posd2'] = np.where((prepost['sample_class'] == 'L_posd2'), prepost[f], np.nan)
            prepost[f + '_posd3'] = np.where((prepost['sample_class'] == 'L_posd3'), prepost[f], np.nan)
            prepost[f + '_posd4'] = np.where((prepost['sample_class'] == 'L_posd4'), prepost[f], np.nan)
    prepost = prepost.groupby('Entry').agg(lambda x: get_mean(x))
    return prepost


def impute_features(df, mode):
    """
    Impute the collected features
    @param df: dataframe
    @param mode: post or pre data mode
    @return: scaled features
    """
    if 'sample_class' in df:
        df = df.drop(columns=['sample_class'])
    variables = {}
    if mode is 'pre':
        variables = var_values.values_pre
    elif mode is 'post':
        variables = var_values.values_post

    entry = df.index
    columns = df.columns
    for feature in variables.keys():
        var = variables[feature]
        if feature in df:
            df[feature] = np.where((df[feature] > var['lower_threshold']), df[feature], np.nan)
            df[feature] = np.where((df[feature] < var['upper_threshold']), df[feature], np.nan)

    scaled_features = fit_iterative_imputer(df, entry, columns)
    return scaled_features


def fit_iterative_imputer(df, entry, columns):
    imp = IterativeImputer(max_iter=10, random_state=0)
    mat = imp.fit_transform(df.values)
    scaled_features = pd.DataFrame(index=entry, columns=columns, data=mat)
    return scaled_features


def impute(df):
    scaler = MinMaxScaler()
    data_rescaled = scaler.fit_transform(df)
    imp = IterativeImputer(max_iter=10, random_state=0)
    mat = imp.fit_transform(data_rescaled)
    return mat


def pca_analysis(data):
    if 'sample_class' in data:
        data = data.drop(columns=['sample_class'])
    data_rescaled = impute(data)
    principal_components = PCA(n_components=2).fit_transform(data_rescaled)
    principal_df = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2'])
    principal_df['Entry'] = data.reset_index(0)['Entry']
    return principal_df


def one_hot_encoder(data, column):
    data[column] = pd.Categorical(data[column])
    dummies = pd.get_dummies(data[column], prefix=column)

    column_position = data.columns.get_loc(column)
    df1 = data.iloc[:, :column_position]
    df2 = data.iloc[:, column_position + 1:]

    df = pd.concat([df1, dummies, df2], axis=1)
    return df


def add_euroscore(data, data_ident):
    spike_cols = [col for col in data_ident.columns if 'Eur_' in col]
    print(data_ident)
    data_ident = data_ident[spike_cols]
    for col in spike_cols:
        if data_ident[col].dtype != float:
            if ('n' or 'j') in data_ident[col].unique():
                data_ident[col] = data_ident[col].replace({'n': 0, 'j': 1})
            else:
                print(col)
                data_ident = one_hot_encoder(data_ident, col)
    # data_ident = data_ident.set_index('Entry')
    data_ident = fit_iterative_imputer(data_ident, data_ident.index, data_ident.columns)
    data = pd.merge(data, data_ident, on='Entry')
    print(spike_cols)
    return data


def preprocessing(mode, data='total', euroscore=False):
    print('[INFO] Preprocessing started for ' + mode + ' data...')
    ident_df = extract_features_ident(data, from_saved=True)
    df = extract_features(mode, data)
    print('[INFO] Extracted features...')
    if data != 'random500':
        df = calculate_creat_clearance(df, ident_df)
        print('[INFO] Calculated eCCR...')
        df = calculate_periods(df, mode)
        print('[INFO] Calculated periods...')
    df = impute_features(df, mode)
    print('[INFO] Imputed features...')
    print(df)
    name = path / ('data/' + mode + '_' + data + '_data.csv')
    df.to_csv(name)
    print('[INFO] Saving to file...')
    if mode == 'pre':
        print('[INFO] Performing PCA...')
        df = pca_analysis(df)
        df.to_csv(path / ('data/' + data + '_pca.csv'))
    else:
        if euroscore:
            df = add_euroscore(df, ident_df)
            df.to_csv(path / ('data/post_' + data + '_with_euroscore.csv'))
        else:
            df.to_csv(path / ('data/post_' + data + '.csv'))
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocessing of post-oeprative data")
    parser.add_argument('--dataset_type', default='total',
                        help='Dataset to use: random500 or total')
    args = parser.parse_args()
    dataset_type = args.dataset_type

    # create new dataframe for pre- and/or post-
    preprocessing('pre', data=dataset_type)
    preprocessing('post', data=dataset_type)
