import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# one-hot encode categorical variables
def get_features_onehot_encoded(features, categorical_vars):
    enc = OneHotEncoder()
    X = features[categorical_vars]
    enc.fit(X)
    features_transformed = enc.transform(X).toarray()
    encoded = pd.DataFrame(features_transformed)

    # append one-hot encoded features, and drop original
    # categorical variables
    features_encoded = pd.concat([features, encoded], axis=1).drop(\
            categorical_vars, axis=1)
    
    # convert all column names to strings
    features_encoded.columns = features_encoded.columns.astype(str)

    return features_encoded

# ordinal encode categorical variables
def get_features_ordinal_encoded(features, categorical_vars):
    enc = OrdinalEncoder()
    X = features[categorical_vars]
    enc.fit(X)
    features_transformed = enc.transform(X)
    encoded = pd.DataFrame(features_transformed)

    # append ordinal encoded features, and drop original
    # categorical variables
    features_encoded = pd.concat([features, encoded], axis=1).drop(\
            categorical_vars, axis=1)
    
    # convert all column names to strings
    features_encoded.columns = features_encoded.columns.astype(str)

    return features_encoded
