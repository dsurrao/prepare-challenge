import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from datetime import datetime
from feature_encoder import get_features_encoded

def make_prediction(train_features, train_labels, test_features):
    # 'year' and 'composite_score' columns will be added to return value
    train_features_and_labels = train_features.merge(train_labels, on='uid')

    # add 'year' column to test features
    prediction_years = pd.DataFrame({ 'year': [2016, 2021] })
    test_features_and_years = test_features.join(prediction_years, how='cross')

    # ignore 'composite_score' in first argument
    exclude_columns = ['composite_score']
    features = pd.concat([\
        train_features_and_labels[\
            [col for col in train_features_and_labels.columns\
                if col not in exclude_columns]],\
        test_features_and_years]).reset_index()

    # create transformed year columns
    composite_score_year = 2021 # date of latest label values
    year_columns = ['rjob_end_03', 'rjob_end_12', 'a16a_12']
    for col in year_columns:
        features[f'{col}_years_since'] = features[col].apply(\
            years_since, args=(composite_score_year, ))
    features.drop(year_columns, axis=1, inplace=True)

    # one hot encode categorical variables
    categorical_vars = get_categorical_vars()
    features_encoded = get_features_encoded(features, categorical_vars)
    # skip new index column and 'uid' column
    exclude_columns = ['index', 'uid']
    features_encoded = features_encoded[\
        [col for col in features_encoded.columns if col not in exclude_columns]]
    
    # remove low variance features
    probability_of_same_value = 0.7
    sel = VarianceThreshold(\
        threshold=(probability_of_same_value * (1 - probability_of_same_value)))
    selected_features_encoded = pd.DataFrame(
        sel.fit_transform(features_encoded))
    
    # build features
    train_features_encoded = selected_features_encoded[:len(train_features_and_labels)]
    train_features_processed = train_features_encoded
    train_features_processed.columns = train_features_processed.columns.astype(str)

    # labels
    train_labels_processed = train_features_and_labels['composite_score']

    # pass train features and labels to classifier. this classifier accepts
    # NaN values.
    est = HistGradientBoostingRegressor().fit(\
        train_features_processed, train_labels_processed)

    # make predictions
    test_features_encoded = selected_features_encoded[len(train_features_and_labels):]
    predicted_labels = est.predict(test_features_encoded)

    # join uid to scores and return
    predicted_labels_rounded = [round(l) for l in predicted_labels]
    test_features_scored = test_features_and_years.copy()
    test_features_scored['composite_score'] = predicted_labels_rounded
    predicted_scores = test_features_scored[['uid', 'year', 'composite_score']]

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

def years_since(year, composite_score_year) -> int | None:
    if year != None:
        return composite_score_year - year
