from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model (already includes preprocessing pipeline)
with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Column order for HTML form (must match training features)
feature_names = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income',
    'rooms_per_household', 'bedrooms_per_room', 'population_per_household',
    'ocean_proximity'
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()

        # Convert numeric values to float
        for key in form_data:
            if key != "ocean_proximity":  # leave category as string
                form_data[key] = float(form_data[key])

        # Create DataFrame for prediction
        input_df = pd.DataFrame([form_data])

        # Predict
        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)

        return render_template(
            "index.html",
            prediction_text=f"Estimated House Price: ${prediction:,.2f}"
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
