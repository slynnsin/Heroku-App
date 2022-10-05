# import libraries
import numpy as np
from flask import Flask, request,render_template
import pickle

# create flask application
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# create end points
# end point for initial application page with text boxes for input
@app.route('/')
def home():
    return render_template('index.html')

# end point for after input is submitted, producing prediction / output
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Salary should be $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)