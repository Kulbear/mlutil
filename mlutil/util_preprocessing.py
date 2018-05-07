# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


def standarize(train, test):
    """ Standarize the dataset.

    Fit a StandardScaler with the training set then applies the same transformation
    on both the training and test set. Return scaled training and test set.
    
    Parameters
    ----------
    train: array-like, shape (nb_samples, nb_features)
        The training set data.

    test: array-like, shape (nb_samples, nb_features)
        The test set data.
    """

    scaler = StandardScaler()
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)

    return scaler, scaled_train, scaled_test


def generate_time_series(df, lookback_step=2):
    """ Generate time-series dataset based on give dataset.

    A time series is a series of data points indexed (or listed or graphed) in time order.
    This function helps on generating time-series data.
    Suppose the input pandas DataFrame has three columns, namely x1, x2, and y,
    the with a lookback_step set to 2, the returned DataFrame will have 
    the following columns x1_t-2, x2_t-2, y_t-2, x1_t-1, x2_t-1, y_t-1,
    x1, x2, and y. 

    Parameters
    ----------
    df: pandas.DataFrame
        A pandas DataFrame that will be used for generating time-seires data. 
        Note the target column is supposed to be the last column.

    lookback_step: int, default 2
        The look-back step size for generating the time-series data. With a
        lookback_step of 2, two steps will be looked back. That is, with current time
        step as t, the features and target at time t-2 and t-1 will be created 
        as new features for time t.

    TODOs
    -----
    1. Support arbitrary target column position/index.
    2. Support padding to avoid NaNs appear in the returned data frame.
    3. Error handlings.

    """

    target_col = df.columns[-1]
    processed_cols = df.columns
    data = df.as_matrix()

    processed_cols = list(df.columns)
    for i in range(1, lookback_step + 1):
        for col in df.columns:
            processed_cols.append('{}_t-{}'.format(col, i))

    result = []
    for idx in range(lookback_step, data.shape[0] + lookback_step):
        result.append(list(data[idx - lookback_step:idx + 1].flatten()))

    proc_df = pd.DataFrame(result, columns=processed_cols)
    processed_cols.remove(target_col)
    processed_cols.append(target_col)
    proc_df = proc_df[processed_cols]

    return proc_df
