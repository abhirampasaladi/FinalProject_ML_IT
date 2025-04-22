from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and encoders
with open("model.pkl", "rb") as f:
    model, scaler, label_encoders = pickle.load(f)

# Load dataset to get unique dropdown values
df = pd.read_csv("dataset_cars.csv")
df['car_age'] = 2025 - df['year']
categorical_cols = ['manufacturer', 'condition', 'cylinders', 'fuel', 'transmission', 'drive', 'type', 'paint_color']
dropdown_values = {col: sorted(df[col].dropna().unique()) for col in categorical_cols}

# Ensure column order matches the trained model
expected_columns = ['manufacturer', 'condition', 'cylinders', 'fuel', 'odometer', 'transmission', 'drive', 'type',
                    'paint_color', 'car_age']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get input data from form
        features = {}
        for col in categorical_cols:
            features[col] = label_encoders[col].transform([request.form[col]])[0]

        features['odometer'] = float(request.form['odometer'])
        features['car_age'] = 2025 - int(request.form['year'])

        # Convert to DataFrame and enforce correct column order
        input_df = pd.DataFrame([features])
        input_df = input_df.reindex(columns=expected_columns)  # Ensure correct order

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)[0]

        return render_template("result.html", prediction=round(prediction, 2))

    return render_template("index.html", dropdown_values=dropdown_values)

if __name__ == "__main__":
    app.run(debug=True)
