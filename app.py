import pickle

import pandas as pd
from flask import Flask,render_template,request
from flask_cors import cross_origin
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)


@app.route('/',methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
@cross_origin()
def predict_result():
    if request.method == 'POST':
        try:
            air_temp = float(request.form['Air temperature [K]'])
            process_temp = float(request.form['Process temperature [K]'])
            rotational_speed = float(request.form['Rotational speed [rpm]'])
            torque = float(request.form['Torque [Nm]'])
            tool_wear = float(request.form['Tool wear [min]'])
            twf = request.form['TWF']
            hdf = request.form['HDF']
            pwf = request.form['PWF']
            osf = request.form['OSF']
            rnf = request.form['RNF']

            filename = 'finalized_model_for_car.pickle'
            loaded_model = pickle.load(open(filename,'rb'))
            scaler =  StandardScaler()
            arr = [air_temp,process_temp,rotational_speed,torque,tool_wear,twf,hdf,pwf,osf,rnf]

            scaler_arr = scaler.fit_transform([arr])
            prediction = loaded_model.predict(scaler_arr)
            print('prediction is', prediction)
            return render_template('results.html',prediction = round(100*prediction[0]))
        except Exception as e:
            raise Exception()
    else:
        render_template('index.html')






if __name__ == '__main__':
    app.run()


