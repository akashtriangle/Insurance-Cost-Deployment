from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
from custom_transformers import *

app = Flask(__name__)

model = pickle.load(open('Insurance Cost Prediction full pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def INPUT_and_PREDICT():
    input_data = {
    'age': int(request.form.get('age')),
    'sex': request.form.get('sex'),
    'bmi': float(request.form.get('bmi')),
    'children': int(request.form.get('children')),
    'smoker': request.form.get('smoker'),
    'region': request.form.get('region')
    }
    # create dataframe
    df = pd.DataFrame([input_data])

    # predict
    ypred = model.predict(df)
    ypred = ypred[0]
    ypred = np.expm1(ypred)
    return f"Predicted Insurance Cost is {str(ypred)}"

if __name__ == '__main__':
    app.run(debug=True)