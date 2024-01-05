def classify_one(data, dv, model):
    x = dv.transform([data])
    y_pred = model.predict_proba(x)[:, 1]
    classes = ['Setosa', 'Versicolour', 'Virginica']
    predicted_class = classes[int(y_pred[0])]
    return predicted_class

