import pickle
from flask import Flask, jsonify, request
from service import classify_one

app = Flask('iris-classification')

MODEL_TYPES = ['lr', 'svm', 'tree', 'knn']

models = {}
dvs = {}

for model_type in MODEL_TYPES:
    with open(f'../models/{model_type}.pck', 'rb') as f:
        dv, model = pickle.load(f)
        models[model_type] = model
        dvs[model_type] = dv


@app.route('/classify', methods=['POST'])
def classify():
    body = request.get_json()
    prediction = classify_one(body['data'], dvs[body['model_type']], models[body['model_type']])

    print(prediction)
    result = {
        'flower_type': prediction
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
