import pandas as pd
from sklearn import preprocessing

# one-hot encode categorical variables
def get_encoded_features(train_features, test_features):
    enc = preprocessing.OneHotEncoder()
    categorical_vars = ['age_03', 'age_12', 'urban_03', 'urban_12',\
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
    X = pd.concat([train_features[categorical_vars],\
        test_features[categorical_vars]])
    enc.fit(X)
    encoded_features_array = enc.transform(X).toarray()
    encoded_features = pd.DataFrame(encoded_features_array)

    # append one-hot encoded features to train set, and drop original
    # categorical variables
    train_features_encoded =\
        pd.concat([train_features, encoded_features[:len(train_features)]],\
            axis=1).drop(categorical_vars, axis=1)

    test_features_encoded =\
        pd.concat([test_features, encoded_features[len(train_features):]],\
            axis=1).drop(categorical_vars, axis=1)
    
    # convert all column names to strings
    train_features_encoded.columns\
        = train_features_encoded.columns.astype(str)

    test_features_encoded.columns\
        = test_features_encoded.columns.astype(str)
    
    return train_features_encoded, test_features_encoded