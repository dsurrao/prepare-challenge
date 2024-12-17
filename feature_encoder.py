import pandas as pd
from sklearn import preprocessing

# one-hot encode categorical variables
def get_encoded_features(train_features, test_features, categorical_vars):
    enc = preprocessing.OneHotEncoder()
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