# testapi.py

# Common python package imports.
import requests
import numpy as np
import pandas as pd
from sklearn import datasets

# Import from model_api/app/estimators/config.py.
from app.features import FEATURES


def get_feature_dists():
    """Get mean and stdev of features for random data generator."""
    data = datasets.load_boston()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    pdf = df.describe().T[['mean', 'std']]
    dist_dict = {}
    for row in pdf.iterrows():
        dist_dict[row[0]] = tuple(row[1].values)
    return dist_dict


def test_api(feature_names, dist_dict, url='http://0.0.0.0:5000/api'):
    """Test the API by making a call and return the response."""
    params = {}
    for feature in feature_names:
        params[feature] = np.random.normal(*dist_dict[feature])
    print('-' * 80, 'INPUTS', '-' * 80)
    print(params)
    response = requests.get(url=url, json=params)
    data = response.json()
    print('-' * 80, 'OUTPUT', '-' * 80)
    print(data)
    return data


if __name__ == '__main__':
    dist_dict = get_feature_dists()
    ### Run locally outside of docker
    url = 'http://0.0.0.0:5000/api'
    for _ in range(1):
        resp = test_api(FEATURES, dist_dict, url=url)
