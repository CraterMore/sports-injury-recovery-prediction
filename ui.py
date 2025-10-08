import pandas as pd
from flask import Flask, render_template, request, jsonify

from regression import InjuryRecoveryPredictor

app = Flask(__name__)

# load trained model
predictor = InjuryRecoveryPredictor()
predictor.load_model("injury_recovery_model.pkl")

# form page
@app.route("/", methods=["GET"])
def ui():
    return render_template("application.html")

@app.route("/predict", methods=["POST"])
def predict():
    # get inputs from form
    age = float(request.form.get("age"))
    gender = request.form.get("gender").strip().title()
    minutes = float(request.form.get("minutes"))
    injury = request.form.get("injury").strip().title()
    body_part = request.form.get("body-part").strip().title()
    severity = request.form.get("severity").strip().title()
    cause = request.form.get("cause").strip().title()
    recurrent = request.form.get("recurrent").strip().title()
    sport = request.form.get("sport").strip().title()
    competition = request.form.get("competition").strip().title()
    region = request.form.get("region").strip().title()
    event = request.form.get("event").strip().title()
    surface = request.form.get("surface").strip().title()

    # input dict matching model features
    input_data = {
        'Age': age,
        'Gender': gender,
        'MinutesPlayedLast30Days': minutes,
        'InjuryType': injury,
        'BodyPart': body_part,
        'Severity': severity,
        'InjuryCause': cause,
        'IsRecurrentInjury': recurrent,
        'Sport': sport,
        'CompetitionLevel': competition,
        'Region':region,
        'EventType': event,
        'SurfaceType': surface,
        'Outcome': 'Fully Recovered', #default/fallback
        'TreatmentMethod': 'Rest', #default
        'CostOfTreatmentEuros': 1800.0 # default, average across test data
    }

    # convert input into DataFrame and match training features
    input_df = pd.DataFrame([input_data])[predictor.feature_columns_reg]

    # apply label encoders to categorical columns
    for col, encoder in predictor.label_encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col].astype(str))

    # scale features
    input_scaled = predictor.scaler.transform(input_df)

    # linear regression prediction -> recovery time (days)
    recovery_time = round(predictor.model_reg.predict(input_scaled)[0])

    # Logistic regression prediction -> probability of full recovery
    # Note: assumes binary labels {0,1} where 1 = Fully Recovered
    if hasattr(predictor.model_clf, "predict_proba"):
        full_recovery_prob = predictor.model_clf.predict_proba(input_scaled)[0][1]
    else:
        full_recovery_prob = predictor.model_clf.predict(input_scaled)[0]

    return render_template("result.html", recovery_time=recovery_time,
                           treatment_method=round(full_recovery_prob * 100, 2))

if __name__ == "__main__":
    app.run(debug=True)