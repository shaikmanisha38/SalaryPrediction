from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved dictionary (model + label_encoders + scaler)
with open("Model/Software Industry Salary Prediction.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
label_encoders = saved["label_encoders"]
scaler = saved["scaler"]

def encode_label(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        # For unseen categories, return a default value (0)
        return 0

@app.route("/")
def index():
    return render_template("index.html")   # Landing page

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Collect form data
            company_name = request.form["company_name"]
            job_title = request.form["job_title"]
            location = request.form["location"]
            job_roles = request.form["job_roles"]
            employment_status = request.form["employment_status"]
            salaries_reported = int(request.form["salaries_reported"])
            rating = float(request.form["rating"])

            # Encode categorical variables using saved LabelEncoders
            company_name_encoded = encode_label(label_encoders['Company Name'], company_name)
            job_title_encoded = encode_label(label_encoders['Job Title'], job_title)
            location_encoded = encode_label(label_encoders['Location'], location)
            job_roles_encoded = encode_label(label_encoders['Job Roles'], job_roles)
            employment_status_encoded = encode_label(label_encoders['Employment Status'], employment_status)

            # Scale numeric features (Rating and Salaries Reported)
            scaled = scaler.transform([[rating, salaries_reported]])
            rating_scaled = scaled[0][0]
            salaries_scaled = scaled[0][1]

            # Prepare final input array
            input_features = np.array([[rating_scaled, company_name_encoded, job_title_encoded,
                                        location_encoded, employment_status_encoded, job_roles_encoded,
                                        salaries_scaled]])

            # Predict salary
            prediction = model.predict(input_features)[0]
            prediction = round(prediction, 2)

            return render_template("result.html", prediction=prediction)

        except Exception as e:
            return f"Error: {e}"

    # GET request → show the prediction form
    return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True)
