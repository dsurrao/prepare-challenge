import pandas as pd
import numpy as np
import os
from average_score_method import make_prediction

if __name__ == '__main__':
    # read input
    train_features = pd.read_csv(os.path.join('data', 'train_features.csv'))
    train_labels = pd.read_csv(os.path.join('data', 'train_labels.csv'))
    test_features = pd.read_csv(os.path.join('data', 'test_features.csv'))

    # make predictions
    predicted_scores = make_prediction(train_features, train_labels,\
        test_features)
    
    # prepare submission file
    submission_format = pd.read_csv(os.path.join('data', 'submission_format.csv'))\
        [['uid', 'year']]
    submission = submission_format.merge(predicted_scores, on='uid')\
        [['uid', 'year', 'composite_score']]

    # write score without decimal place
    submission.to_csv(os.path.join('data', 'submission.csv'),\
        float_format='%.0f', index=False)
