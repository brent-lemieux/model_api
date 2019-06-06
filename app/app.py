from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np
from estimators.config import MODEL_CONFIG


app = Flask(__name__)
app.secret_key = 'something_secret'

MODEL = pickle.load(open('estimators/model.pkl', 'rb'))
FEATURES = MODEL_CONFIG['feature_names']


@app.route('/')
def docs():
    """Describe the model API inputs and outputs for users."""
    return render_template('docs.html')


@app.route('/api', methods=['GET'])
def api():
    """Handle request and output model score in json format."""
    if not request.json:
        return jsonify({'error': 'no request received'})
    # Parse request args and format into feature array for prediction.
    x_list, missing_data = parse_args(request.json)
    x_array = np.array([x_list])
    # Predict on features provided and return response in JSON.
    estimate = int(MODEL.predict(x_array)[0])
    response = dict(ESTIMATE=estimate, MISSING_DATA=missing_data)
    return jsonify(response)


def parse_args(request_dict):
    """Parse model features from incoming requests formatted in JSON.

    Arguments:
        request_dict (dict) - contains request information.

    Returns:
        x_list (list) - ordered features for model
        missing_data (bool) - flag informing API user whether or not features
            were missing from their request.
    """
    missing_data = False
    x_list = []
    # Iterate through the features list and append to x_list in correct order.
    for feature in FEATURES:
        value = request_dict.get(feature, None)
        if value:
            x_list.append(value)
        else:
            # If the feature is missing, append a 0 and notify of missing data.
            x_list.append(0)
            missing_data = True
    return x_list, missing_data



if __name__ == '__main__':
    # For testing purposes.
    app.run(host='0.0.0.0', debug=True)
