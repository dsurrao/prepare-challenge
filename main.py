import pandas as pd
import numpy as np
import os
from sklearn.ensemble import HistGradientBoostingRegressor
from feature_encoder import get_encoded_features

if __name__ == '__main__':
    train_features = pd.read_csv(os.path.join('data', 'train_features.csv'))
    test_features = pd.read_csv(os.path.join('data', 'test_features.csv'))
    train_labels = pd.read_csv(os.path.join('data', 'train_labels.csv'))

    # compute average composite score for each person, and join that to the 
    # training set
    averaged_score = train_labels.groupby("uid")[["composite_score"]]\
        .mean().reset_index()
    train_features_with_score = train_features.set_index('uid').join(\
        averaged_score.set_index('uid')).reset_index()
    
    # skip 'composite_score' for encoding
    train_features_encoded, test_features_encoded =\
        get_encoded_features(\
            train_features_with_score[train_features_with_score.columns[:-1]],\
            test_features)

    # skip 'uid' for features
    train_features_processed = train_features_encoded[\
        train_features_encoded.columns[1:]]
    train_labels_processed = train_features_with_score['composite_score']

    # this classifier accepts NaN values,
    est = HistGradientBoostingRegressor().fit(\
        train_features_processed, train_labels_processed)

    # make predictions, skip 'uid' for features
    test_features_processed = test_features_encoded[test_features_encoded.columns[1:]]
    predicted_labels = est.predict(test_features_processed)

    # round labels to integer values
    predicted_labels_int = [round(l) for l in predicted_labels]

    # prepare submission file
    predicted_scores = pd.DataFrame({ 'composite_score': predicted_labels_int })
    submission_output = pd.concat([test_features_encoded, predicted_scores], axis=1)
    submission_format = pd.read_csv(os.path.join('data', 'submission_format.csv'))\
        [['uid', 'year']]
    submission = submission_format.merge(submission_output, on='uid')\
        [['uid', 'year', 'composite_score']]

    # write score without decimal place
    submission.to_csv(os.path.join('data', 'submission.csv'),\
        float_format='%.0f', index=False)
