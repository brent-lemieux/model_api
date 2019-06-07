# model_engine.py

# Common python package imports.
import pickle
import numpy as np
import pandas as pd
from sklearn import datasets, ensemble

# Import from model_api/app/app.py.
from app.features import FEATURES


def model_pipeline(feature_names):
    """Get the data, train the model, and save it."""
    X, y = get_model_data(feature_names)
    model = ensemble.RandomForestRegressor(n_estimators=50)
    model.fit(X, y)
    pickle.dump(model, open('app/model.pkl', 'wb'))
    print('Success!')


def get_model_data(feature_names):
    """Load and prepare data for modeling."""
    data = datasets.load_boston()
    df = build_dataframe(data)
    df = clean_data(df)
    # Limit the feature set for simplicity.
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
    model_pipeline(FEATURES)
