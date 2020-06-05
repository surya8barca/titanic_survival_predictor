from flask import Flask,jsonify,request
import joblib
from flask.templating import render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('titanic.html')

@app.route("/predict/",methods=['GET'])
def predict():
    result=request.args
    print(result)
    model=joblib.load('logistic_regression_model.sav')
    if(result["embark"]=='S'):
        data=[[int(result["pclass"]),int(result['age']),int(result['nosiblings']),int(result['noparents']),float(result["fare"]),int(result["gender"]),0,1]]
    else:
        data=[[int(result["pclass"]),int(result['age']),int(result['nosiblings']),int(result['noparents']),float(result["fare"]),int(result["gender"]),1,0]]
    prediction=model.predict(data)
    survival=''
    if(prediction[0]==0):
        survival='Could Not Survive'
    else:
        survival='Survived'
    return jsonify({'prediction': survival})

if __name__ == '__main__':
    app.run()