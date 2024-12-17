import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from feature_encoder import get_encoded_features

def make_prediction(train_features, train_labels, test_features):
    # compute average composite score for each person
    averaged_score = train_labels.groupby("uid")[["composite_score"]]\
        .mean().reset_index()

    # one hot encode categorical variables
    categorical_vars = get_categorical_vars()
    train_features_encoded, test_features_encoded =\
        get_encoded_features(train_features, test_features, categorical_vars)
    
    # remove low variance features, ignore 'uid'
    all_data_encoded = pd.concat([
        train_features_encoded[train_features_encoded.columns[1:]],\
        test_features_encoded[test_features_encoded.columns[1:]]
    ])
    probability_of_same_value = 0.7
    sel = VarianceThreshold(\
        threshold=(probability_of_same_value * (1 - probability_of_same_value)))
    selected_features_encoded = pd.DataFrame(sel.fit_transform(all_data_encoded))

    # build features and labels joined to average composite score
    train_features_encoded_with_uid = pd.concat(\
        [selected_features_encoded[:len(train_features)],
        train_features['uid']],
        axis=1)

    train_labels_processed = train_features_encoded_with_uid.merge(\
        averaged_score, on='uid')['composite_score']
    
    train_features_processed = train_features_encoded_with_uid.drop(['uid'],\
        axis=1)
    train_features_processed.columns = train_features_processed.columns.astype(str)

    # this classifier accepts NaN values,
    est = HistGradientBoostingRegressor().fit(\
        train_features_processed,\
        train_labels_processed)

    # make predictions, skip 'uid' for features
    test_features_processed = selected_features_encoded[len(train_features_encoded):]
    predicted_labels = est.predict(test_features_processed)
    predicted_labels_rounded = [round(l) for l in predicted_labels]

    scores = pd.DataFrame({ 'composite_score': predicted_labels_rounded })
    predicted_scores = pd.concat([test_features, scores], axis=1)\
        [['uid', 'composite_score']]

    return predicted_scores

def get_categorical_vars():
    return ['age_03', 'age_12', 'urban_03', 'urban_12',\
        'married_03', 'married_12', 'edu_gru_03', 'edu_gru_12',\
        'n_living_child_03',\
        'n_living_child_12', 'glob_hlth_03', 'glob_hlth_12', 'bmi_03',\
        'bmi_12', 'decis_famil_03', 'decis_famil_12', 'decis_personal_03',\
        'decis_personal_12', 'employment_03', 'employment_12',\
        'satis_ideal_12',\
        'satis_excel_12', 'satis_fine_12', 'cosas_imp_12',\
        'wouldnt_change_12',\
        'memory_12', 'ragender', 'rameduc_m', 'rafeduc_m', 'sgender_03',\
        'sgender_12', 'rjlocc_m_03', 'rjlocc_m_12', 'rjobend_reason_03',\
        'rjobend_reason_12', 'rrelgimp_03', 'rrelgimp_12', 'rrfcntx_m_12',\
        'rsocact_m_12', 'rrelgwk_12', 'a22_12', 'a33b_12', 'a34_12', 'j11_12']