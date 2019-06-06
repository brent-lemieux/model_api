import pickle
import numpy as np
import pandas as pd
from sklearn import datasets, ensemble
from app.estimators.config import MODEL_CONFIG


def model_pipeline(feature_names):
    """Get the model data, train the model, and save it.

    The model is stored in .pkl format in `app/estimators/` for use by the
    Model API defined in `app/app.py`.

    Arguments:
        feature_names (list) - names of features to be used in model; must be
            lowercased versions of features contained in the sklearn Boston
            house prices dataset
    """
    X, y = get_model_data(feature_names)
    model = ensemble.RandomForestRegressor(n_estimators=50)
    model.fit(X, y)
    pickle.dump(model, open('app/estimators/model.pkl', 'wb'))
    print('Success!')


def get_model_data(feature_names):
    """Load and prepare data for modeling."""
    data = datasets.load_boston()
    df = build_dataframe(data)
    df = clean_data(df)
    X, y = df[feature_names].values, df['target'].values
    return X, y


def build_dataframe(data):
    """Build dataframe to facilitate cleaning."""
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df


def clean_data(df):
    """Clean data in preparation for modeling."""
    df = df.loc[df['target'] != 50].copy()
    return df


if __name__ == '__main__':
    feature_names = MODEL_CONFIG['feature_names']
    model_pipeline(feature_names)
