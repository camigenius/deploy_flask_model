import numpy as np
from flask import Flask,request,json,render_template
import pickle


# creamos la app de flask
app = Flask(__name__)
model = pickle.load(open("model.pkl",'rb'))

@app.route("/")
def Home():
    return render_template('index.html')

@app.route("/predict",methods=["POST"])
def predict():
    float_features = [float(x)for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html",prediction_text= "El Valor en miles de Dolares de la casa es : {}".format(prediction))


if __name__ == "__main__":
    app.run(debug=True)