from flask import Flask, escape, request, render_template

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        data=request.form['essay']
        return render_template('predict.html',data=data)
if __name__=='__main__':
	app.run(debug=True)