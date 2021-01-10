from flask import Flask, request, jsonify, render_template
import numpy as np


app = Flask(__name__, static_url_path='')
app.config["DEBUG"]=True


@app.before_first_request
def load_model_to_app():
    app.predictor = load_model('./static/model/model.h5')

@app.route("/")
def index():
    return render_template('index.html', pred = 0)

@app.route('/predict', methods=['POST'])
def predict():
    data = [request.form['spatial_length'],
            request.form['spatial_width'],
            request.form['petal_length'],
            request.form['petal_width']]

    data = np.array([np.asarray(data, dtype=float)])

    predictions = app.predictor.predict(data)
    print('INFO Predictions: {}'.format(predictions))

    class_ = np.where(predictions == np.amax(predictions, axis=1))[1][0]

    return render_template('index.html', pred=class_)

def main():
    """Run the app."""
    app.run(host='0.0.0.0', port=5000, debug=True)  # nosec

if __name__ == '__main__':
    main()