import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # For POST request
    if request.method == 'POST':
        print("[INFO] Loading model...")

        with open("fdemand.pkl", "rb") as f:
            model = pickle.load(f)

        y = request.form.values()
        input_features = []
        print(y)
        for x in y:
            if x.isnumeric():
                print(x)
                input_features.append(float(x))

        print(input_features)
        features_value = [np.array(input_features)]
        print(features_value)

        prediction = model.predict(features_value)
        output = round(prediction[0])
        print(output)
        return render_template('predict.html', title="Predict", prediction_text=output)

    return render_template('predict.html', title="Predict")  # , prediction_text=output)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True)
