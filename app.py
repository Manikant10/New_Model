from flask import Flask, render_template
app = Flask(__name__)
import joblib
import pandas as pd
import numpy as np

# Load the saved model
model = joblib.load('loan1.pkl')

app = Flask(__name__)

# Homepage
@app.route('/')
def home():
    return render_template('index.html')

# Predict function for loan approval prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Getting input data from form
        gender = request.form['gender']
        married = request.form['married']
        dependents = request.form['dependents']
        self_employed = request.form['self_employed']
        applicant_income = request.form['applicant_income']
        coapplicant_income = request.form['coapplicant_income']
        loan_amount = request.form['loan_amount']
        loan_amount_term = request.form['loan_amount_term']
        credit_history = request.form['credit_history']
        property_area = request.form['property_area']
        
        # Convert the input data to a DataFrame and perform encoding
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_amount_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area]
        })
        
        # Encoding categorical variables
        input_data = pd.get_dummies(input_data)
        
        # Making predictions using the trained model
        prediction = model.predict(input_data)
        
        # Returning result to user
        if prediction == 1:
            return render_template('index.html', prediction_text="Loan Approved!")
        else:
            return render_template('index.html', prediction_text="Loan Denied!")
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
