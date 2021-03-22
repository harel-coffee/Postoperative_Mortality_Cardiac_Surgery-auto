import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from preprocessing import var_values

path = Path(__file__).parent.parent

min_max_scaler = MinMaxScaler(feature_range=(0, 1))


# --------- Preprocessing of intraoperative data ----------

def read_intra():
    """
    read all CORON intraoperative datasets and combine them into a single dataframe
    @return: resulting dataframe
    """

    surgeries = ['CORON', 'KLCOR', 'AOKLP', 'XKLEP']
    dates = ['1994', '1999', '2009']
    li = []
    for surgery in surgeries:
        for date in dates:
            dataset_name = path / ('data/' + surgery + '_' + date + '_ok.txt')
            dataset = pd.read_csv(dataset_name)
            dataset['Entry'] = dataset['Entry'].astype(str) + date + '_' + surgery
            li.append(dataset)

    ok = pd.concat(li, axis=0, ignore_index=True)
    return ok


def read_intra_random500():
    """
    read random500 intraoperative datasets and combine them into a single dataframe
    @return: combined dataframe
    """
    ok1 = pd.read_csv(path / 'data500/ok1.txt')
    ok2 = pd.read_csv(path / 'data500/ok2.txt')
    ok3 = pd.read_csv(path / 'data500/ok3.txt')
    ok4 = pd.read_csv(path / 'data500/ok4.txt')
    ok5 = pd.read_csv(path / 'data500/ok5.txt')

    li = [ok1, ok2, ok3, ok4, ok5]

    ok = pd.concat(li, axis=0, ignore_index=True)
    return ok


def get_start(x, reltime):
    """
    get the starting point of perfusion/anesthesie/operatie
    @param x: corresponding column of the dataset
    @param reltime: reltime
    @return: starting point of perfusion/anesthesie/operatie
    """
    idx = x.first_valid_index()
    if idx is None:
        return np.nan
    return reltime.iloc[idx] if x[idx] == ['Ja'] else np.nan


def get_end(x, reltime):
    """
    get the last block of perfusion/anesthesie/operatie
    @param x: corresponding column of the dataset
    @param reltime: reltime
    @return: last block of perfusion/anesthesie/operatie
    """
    idx = x.last_valid_index()
    if idx is None:
        return np.nan
    return reltime.iloc[idx] if x[idx] == ['Nee'] else np.nan


def get_first_block(x, reltime):
    """
    get the first block of perfusion/anesthesie/operatie
    @param x: corresponding column of the dataset
    @param reltime: reltime
    @return: first block of perfusion/anesthesie/operatie
    """
    idx = np.array(x[x.notna()].index)
    for i in range(idx.size - 1):
        if x[idx[i]] == 'Ja' and x[idx[i + 1]] == 'Nee':
            return [reltime[idx[i] - 10], reltime[idx[i + 1]]]

    return [np.nan, np.nan]


def filter_outliers(column, var):
    """
    filter outliers based on the provided thresholds
    @param column: column to filter in the dataset
    @param var: variable to filter in the dataset
    @return:
    """
    var = var_values.variables[var]
    median_vals = column.rolling(var['window']).median()
    difference = column - median_vals
    # oob = np.where(np.logical_or((column < var['lower_lim']),(column > var['upper_lim'])),1,0)
    column.loc[(np.abs(difference) > var['threshold'])] = np.NaN
    filtered_col = column.interpolate(method='linear')
    return filtered_col


def normalize_column(series, scaler=min_max_scaler):
    """
    Normalized the data
    @param series: time series
    @param scaler: type of scaler
    @return:
    """
    series = series.values.reshape(-1, 1)
    s_scaled = scaler.fit_transform(series)
    s_scaled = s_scaled.reshape((-1))
    return s_scaled


def filtering(data='total', features='hemodynamics', name=None):
    """
    filtering the data from outliers, fixing heart rate values and missing values interpolation
    @param name: name of dataframe to save
    @param features: hemodynamics or temperature
    @param data:
        'random500' - use random500 dataset
        else - use whole CORON dataset
    @return: resulting dataframe
    """
    if data == 'random500':
        df = read_intra_random500()
        print('[INFO] Read random500 patients...')
    else:
        df = read_intra()
        print('[INFO] Read all patients...')
    df = filter_operation_period(df, features)
    print('[INFO] Filtered operation period...')

    # interpolate missing values
    df = df.interpolate(method='linear')
    print('[INFO] Interpolated missing values...')

    var = list(df)
    var.remove('Entry')

    for v in var:
        df[v] = df.groupby('Entry')[v].transform(lambda x: filter_outliers(x, v))
        df[v] = normalize_column(df[v], min_max_scaler)
    print('[INFO] Normalized and scaled the data...')

    file_name = features if name is None else name
    df.to_csv(path / ('data/' + file_name + '.csv'))
    print('[INFO] Saved to file...')
    return df


def percentage_no_perfusie(df):
    ids = list(df['Entry'].unique())
    count = 0
    for file in ids:
        sub_df = df.loc[df['Entry'] == file]
        # interpolate missing values
        #     sub_df = sub_df.interpolate(method = 'linear')
        sub_df_perfusie = sub_df.loc[df['Perfusie'].notna()]
        if sub_df_perfusie.empty:
            count += 1

    print(count / len(ids))


def filter_operation_period(ok, features='total'):
    """
    filters the data based on perfusion times
    @param ok: outliers-free dataframe
    @param features: the final type of resulting dataframe
        ('perfusions' - 3 dataframes for perfusion times
         'total' - total dataframe
         'hemodynamics' - include only  'PArtM', 'PArtD', 'PArtS', 'PCVD', 'HR',
         'CETCO2', 'Pulsox'
          'temperature' - include only 'Tcentraal', 'Tnp', 'Thuid')
    @return: resulting dataframe
    """
    # divide heart rate column values by 2 because it is doubled
    ok.loc[:, 'HR'] = ok['HR'].apply(lambda x: x / 2)
    ok.loc[:, 'Pulsox'] = ok['Pulsox'].apply(lambda x: x / 2)

    ok_temp = ok[
        ['Entry', 'Reltime', 'Tcentraal', 'Tnp', 'Thuid', 'PArtM', 'PArtD', 'PArtS', 'THLMArt', 'THLMVen', 'PCVD', 'HR',
         'CETCO2', 'Pulsox', 'Perfusie', 'Operatie periode']]

    ok_temp['Anesthesie'] = [re.findall('(?<=Anesthesie_)(Ja|Nee)', i) if isinstance(i, str) else np.nan for i in
                             ok_temp['Operatie periode']]
    ok_temp['Operatie'] = [re.findall('(?<=Operatie_)(Ja|Nee)', i) if isinstance(i, str) else np.nan for i in
                           ok_temp['Operatie periode']]

    times = pd.DataFrame()

    times['start_anesthesie'] = ok_temp.groupby('Entry')['Anesthesie'].apply(lambda x: get_start(x, ok_temp['Reltime']))
    times['end_anesthesie'] = ok_temp.groupby('Entry')['Anesthesie'].apply(lambda x: get_end(x, ok_temp['Reltime']))

    times['start_operatie'] = ok_temp.groupby('Entry')['Operatie'].apply(lambda x: get_start(x, ok_temp['Reltime']))
    times['end_operatie'] = ok_temp.groupby('Entry')['Operatie'].apply(lambda x: get_end(x, ok_temp['Reltime']))

    times['max_reltime'] = ok_temp.groupby('Entry')['Reltime'].apply(max)

    times['start_perfusie'], times['end_perfusie'] = zip(
        *ok_temp.groupby('Entry')['Perfusie'].apply(lambda x: get_first_block(x, ok_temp['Reltime'])))
    times['start_perfusie'] = times['start_perfusie']
    times['end_perfusie'] = times['end_perfusie']

    times['start'] = [y if np.isnan(x) else x for x, y in zip(times['start_anesthesie'], times['start_operatie'])]
    times['end'] = [y if np.isnan(x) else x for x, y in zip(times['end_anesthesie'], times['end_operatie'])]
    times['end'] = [y if np.isnan(x) else x for x, y in zip(times['end'], times['max_reltime'])]

    times = times.reset_index()

    ok_operations = pd.DataFrame.merge(ok_temp, times, on='Entry')

    # delete patients with no start/end times
    ok_operations = ok_operations.dropna(subset=['start', 'end', 'start_perfusie', 'end_perfusie'])

    # times from float to int
    ok_operations['start'] = ok_operations['start'].astype(int)
    ok_operations['end'] = ok_operations['end'].astype(int)
    ok_operations['start_perfusie'] = ok_operations['start_perfusie'].astype(int)
    ok_operations['end_perfusie'] = ok_operations['end_perfusie'].astype(int)

    # delete time before and after operation
    ok_operations = ok_operations[
        (ok_operations['Reltime'] >= ok_operations['start']) & (ok_operations['Reltime'] <= ok_operations['end'])]

    # temp outside of perfusion = nan
    ok_operations[(ok_operations['Reltime'] < ok_operations['start_perfusie']) & (
            ok_operations['Reltime'] > ok_operations['end_perfusie'])][['Tnp', 'Tcentraal', 'Thuid']] = np.nan

    # Pulsox, HR, CETCO2 nan during perfusion
    ok_operations[(ok_operations['Reltime'] > ok_operations['start_perfusie']) & (
            ok_operations['Reltime'] < ok_operations['end_perfusie'])][['HR', 'CETCO2', 'Pulsox']] = np.nan

    print('number of patients: ', len(ok_operations['Entry'].unique()))
    if features == 'total':
        ok_operations = ok_operations[
            ['Entry', 'Tcentraal', 'Tnp', 'Thuid', 'PArtM', 'PArtD', 'PArtS', 'THLMArt', 'THLMVen', 'PCVD', 'HR',
             'CETCO2', 'Pulsox']]
        return ok_operations
    elif features == 'hemodynamics':
        ok_operations = ok_operations[
            ['Entry', 'PArtM', 'PArtD', 'PArtS',
             'PCVD', 'HR', 'CETCO2', 'Pulsox']]

        return ok_operations
    elif features == 'temperature':
        ok_operations = ok_operations[
            ['Entry', 'Tcentraal', 'Tnp', 'Thuid', 'THLMArt', 'THLMVen']]
        return ok_operations
    elif features == 'perfusions':
        pre_perfusie = ok_operations[ok_operations['Reltime'] < ok_operations['start_perfusie']]
        in_perfusie = ok_operations[(ok_operations['Reltime'] >= ok_operations['start_perfusie']) & (
                ok_operations['Reltime'] <= ok_operations['end_perfusie'])]
        post_perfusie = ok_operations[ok_operations['Reltime'] > ok_operations['end_perfusie']]

        pre_perfusie = pre_perfusie[['Entry', 'PArtM', 'PArtD', 'PArtS', 'PCVD', 'HR', 'CETCO2', 'Pulsox']]
        in_perfusie = in_perfusie[
            ['Entry', 'Tcentraal', 'Tnp', 'Thuid', 'PArtM', 'PArtD', 'PArtS', 'THLMArt', 'THLMVen', 'Pulsox']]
        post_perfusie = post_perfusie[['Entry', 'PArtM', 'PArtD', 'PArtS', 'PCVD', 'HR', 'CETCO2', 'Pulsox']]
        pre_perfusie.to_csv('pre_perf.csv')
        in_perfusie.to_csv('in_perf.csv')
        post_perfusie.to_csv('post_perf.csv')
        return pre_perfusie, in_perfusie, post_perfusie


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocessing of intra-operative data")
    parser.add_argument('--dataset_type', default='total',
                        help='Dataset to use: random500 or total')
    parser.add_argument('--features', default='total',
                        help='Features to use: total, hemodynamics or temperature')
    args = parser.parse_args()
    dataset_type = args.dataset_type
    features = args.features

    # creating a dataframe of hemodynamics
    dataframe = filtering(data=dataset_type, features=features, name=dataset_type)
    print(len(dataframe['Entry'].unique()))
