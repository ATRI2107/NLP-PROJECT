from flask import Flask, escape, request, render_template
import model_train as ml

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('index.html')
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
#       
        essay=request.form['essay']
        score = ml.get_predictions(essay)
        return render_template('predict.html',score=score)
    
if __name__=='__main__':
	app.run(debug=True)