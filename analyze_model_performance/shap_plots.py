import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def load_data(expname, eur=False):
    path_to_data = '../training_data_euroscorei/' if eur else '../training_data_default/'
    train = np.load(path_to_data + expname + '/train.npy', allow_pickle=True)
    test = np.load(path_to_data + expname + '/test.npy', allow_pickle=True)

    x_train = np.load(path_to_data + expname + '/x_train_gb.npy')
    x_test = np.load(path_to_data + expname + '/x_test_gb.npy')
    y_train = np.load(path_to_data + expname + '/y_train_gb.npy')
    y_test = np.load(path_to_data + expname + '/y_test_gb.npy')
    if eur is True:
        model = joblib.load(path_to_data + expname + '/model_xgb_eur.joblib')
    else:
        model = joblib.load(path_to_data + expname + '/model_xgb.joblib')

    y_proba = model.predict_proba(x_test)
    y_proba = y_proba if y_proba.shape[1] <= 1 else y_proba[:, 1]
    proba_name = 'eur' if eur else 'def'
    np.save(path_to_data + expname + '/y_proba_{}.npy'.format(proba_name), y_proba)

    return model, train, test, x_train, x_test, y_train, y_test, y_proba


def get_features(eur):
    features = ['Leukocytes', 'Creatinine', 'LDH', 'Trombocytes', 'ALAT', 'ASAT', 'Urea',
                'Hb', 'Monocytes', 'Lymphocytes', 'Neutrophils', 'Glucose', 'eCCR',
                'Creatinine (6h)', 'Creatinine (12h)', 'Creatinine (24h)', 'Creatinine (48h)',
                'Creatinine (72h)', 'Creatinine (96h)', 'Urea (6h)', 'Urea (12h)',
                'Urea (24h)', 'Urea (48h)', 'Urea (72h)', 'Urea (96h)',
                'LDH (6h)', 'LDH (12h)', 'LDH (24h)', 'LDH (48h)', 'LDH (72h)',
                'LDH (96h)', 'Glucose (6h)', 'Glucose (12h)', 'Glucose (24h)',
                'Glucose (48h)', 'Glucose (72h)', 'Glucose (96h)', 'Hb (6h)',
                'Hb (12h)', 'Hb (24h)', 'Hb (48h)', 'Hb (72h)', 'Hb (96h)',
                'Leukocytes (6h)', 'Leukocytes (12h)', 'Leukocytes (24h)', 'Leukocytes (48h)',
                'Leukocytes (72h)', 'Leukocytes (96h)', 'Trombocytes (6h)', 'Trombocytes (12h)',
                'Trombocytes (24h)', 'Trombocytes (48h)', 'Trombocytes (72h)', 'Trombocytes (96h)',
                'ALAT (6h)', 'ALAT (12h)', 'ALAT (24h)', 'ALAT (48h)',
                'ALAT (72h)', 'ALAT (96h)', 'ASAT (6h)', 'ASAT (12h)',
                'ASAT (24h)', 'ASAT (48h)', 'ASAT (72h)', 'ASAT (96h)',
                'Neutrophils (6h)', 'Neutrophils (12h)', 'Neutrophils (24h)', 'Neutrophils (48h)',
                'Neutrophils (72h)', 'Neutrophils (96h)', 'Monocytes (6h)',
                'Monocytes (12h)', 'Monocytes (24h)', 'Monocytes (48h)',
                'Monocytes (72h)', 'Monocytes (96h)', 'Lymphocytes (6h)',
                'Lymphocytes (12h)', 'Lymphocytes (24h)', 'Lymphocytes (48h)', 'Lymphocytes (72h)',
                'Lymphocytes (96h)', 'eCCR (6h)', 'eCCR (12h)',
                'eCCR (24h)', 'eCCR (48h)', 'eCCR (Д72h)', 'eCCR (96h)']
    eur_features = ['Eur_score', 'Eur_pulm', 'Eur_artp', 'Eur_ndys', 'Eur_psear', 'Eur_kreat',
                    'Eur_endoc', 'Eur_state', 'Eur_angp', 'Eur_LVdys_goed', 'Eur_LVdys_matig/LVEF_30-50%',
                    'Eur_LVdys_slecht/LVEF_<_30%', 'Eur_infar', 'Eur_hyper',
                    'Eur_plan', 'Eur_other', 'Eur_aorta', 'Eur_rupt', 'Eur_krval', 'Eur_eplan_electief',
                    'Eur_eplan_levensbedreigend', 'Eur_eplan_urgent']
    lstm_features = ['f%d' % i for i in range(128)]
    postop_feat = features if eur is False else features + eur_features
    new_features = lstm_features + postop_feat
    return new_features


def plot_shap():
    explainer = shap.Explainer(model, pd.DataFrame(x_train, columns=new_features))
    shap_values = explainer(pd.DataFrame(x_test, columns=new_features))
    raw_operations_names = ident[ident['Entry'].isin(test)]['Operation type'].to_list()
    operations_names = {'CORON': 'CABG', 'KLEP': 'Valve', 'KLCOR': 'Combined'}
    operations = [operations_names[o] for o in raw_operations_names]
    shap.plots.bar(shap_values.cohorts(operations).abs.mean(0), max_display=10, show=False)
    plt.savefig(str('../plots/cohorts_{}.png'.format(mortality[2])), bbox_inches='tight', facecolor='white')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SHAP plots")
    parser.add_argument('--exp_name', help='experiment name')
    parser.add_argument('--data_type', help='data type (post, all, intra or intra-only')
    parser.add_argument('--mortality', help='Mortality (30d, 1y or 5y)')
    parser.add_argument('--eur', default=False, type=bool, help='Euroscore True or False')
    ident = pd.read_csv('../data500/ident.csv', index_col=0)
    args = parser.parse_args()

    folder = args.exp_name
    data = args.data_type
    mortality = args.mortality
    eur = args.eur
    model, train, test, x_train, x_test, y_train, y_test, y_proba = load_data(folder.format(data, mortality),
                                                                              eur=eur)
    new_features = get_features(eur)
    plot_shap()
