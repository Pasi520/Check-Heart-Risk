from flask import Flask, render_template, request 
import joblib as jb
import numpy as np

model = jb.load('heart_risk_regression.sav')
model_poly = jb.load('model_poly.sav')
model_qntl_data = jb.load('model_qntl_data.sav')
model_qntl_target = jb.load('model_qntl_target.sav')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('patient_details.html')

@app.route('/getresults', methods=['POST'])
def getresults():
    
    result = request.form

    gen_dict = {'female':0, 'male':1}
    smoke_dict = {'no':0, 'yes':1}
    bmp_dict = {'no':0, 'yes':1}
    diab_dict = {'no':0, 'yes':1}

    name = result['name']
    gender = (result['gender'])
    age = float(result['age'])
    tc = float(result['tc'])
    hdl = float(result['hdl'])
    smoke = (result['smoke'])
    bmp = (result['bpm'])
    diab = (result['diab'])

    test_data = np.array([gen_dict[gender],age,tc,hdl,smoke_dict[smoke],bmp_dict[bmp],diab_dict[diab]]).reshape(1,-1)

    test_data = model_qntl_data.transform(test_data)
    test_data = model_poly.transform(test_data)

    predict = model.predict(test_data)
    predict = model_qntl_target.inverse_transform(predict)

    result = {"name":name, "risk":round(predict[0][0],2)}

    return render_template('patient_results.html',results=result)
app.run(debug=True)