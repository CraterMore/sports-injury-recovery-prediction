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
    # all possible treatment methods
    all_treatment_methods = ['Massage', 'Surgery', 'Physiotherapy', 'Rest', 'Ice/Heat Therapy', 'Medication']

    # get fixed inputs from form
    age = float(request.form.get("age"))
    gender = to_title_case(request.form.get("gender"))
    minutes = float(request.form.get("minutes"))
    injury = to_title_case(request.form.get("injury"))
    body_part = to_title_case(request.form.get("body-part"))
    severity = to_title_case(request.form.get("severity"))
    cause = to_title_case(request.form.get("cause"))
    recurrent = to_title_case(request.form.get("recurrent"))
    sport = to_title_case(request.form.get("sport"))
    competition = to_title_case(request.form.get("competition"))
    region = to_title_case(request.form.get("region"))
    event = to_title_case(request.form.get("event"))
    surface = to_title_case(request.form.get("surface"))

    # track best outcome
    best_prob = -1.0
    best_treatment = None
    best_recovery_time = None

    # input dict matching model features
    base_input_data = {
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
        'CostOfTreatmentEuros': 1800.0 # default, average across test data
    }

    # iterate through all treatment methods
    for treatment in all_treatment_methods:
        # create new input set for treatment
        current_input_data = base_input_data.copy()
        # change treatment for current iteration
        current_input_data['TreatmentMethod'] = treatment

        # convert input into DataFrame and match training features
        input_df = pd.DataFrame([current_input_data])[predictor.feature_columns_reg]

        # apply label encoders to categorical columns
        # use a temporary copy for transformation
        temp_df = input_df.copy()
        for col, encoder in predictor.label_encoders.items():
            if col in temp_df.columns:
                temp_df[col] = encoder.transform(temp_df[col].astype(str))

        # scale features
        input_scaled = predictor.scaler.transform(temp_df)

        # linear regression prediction -> recovery time (days)
        recovery_time = round(predictor.model_reg.predict(input_scaled)[0])

        # Logistic regression prediction -> probability of full recovery
        # assumes binary labels {0,1} where 1 = Fully Recovered
        if hasattr(predictor.model_clf, "predict_proba"):
            full_recovery_prob = predictor.model_clf.predict_proba(input_scaled)[0][1]
        else:
            full_recovery_prob = predictor.model_clf.predict(input_scaled)[0]

        print("treatment:" + treatment)
        print("probability: " + str(full_recovery_prob))
        # check if this is the best probability found so far
        if full_recovery_prob > best_prob:
            best_prob = full_recovery_prob
            best_treatment = treatment
            best_recovery_time = recovery_time

    return render_template(
        "result.html",
        recovery_time=best_recovery_time,
        treatment=best_treatment,
        full_recovery_prob=round(best_prob, 2)
    )


# Add this function outside of the Flask routes (e.g., just below the imports)

def to_title_case(s):
    """
    Applies title case to a string but handles specific exceptions
    where only the first word should be capitalized.
    """
    if not s:
        return s

    # Strip whitespace first
    s = s.strip()

    # Define exceptions where only the first word should be capitalized
    exceptions = ["Warm-up", "Non-contact"]

    # Check if the string (in any case) matches an exception
    # We check if the lowercased version matches one of the lowercased exceptions
    # The return value is the correctly formatted string
    for exc in exceptions:
        if s.lower() == exc.lower():
            # For "Warm-up" and "Non-contact", we want "Warm-up" and "Non-contact"
            # Since the original form values are lowercase, we return the Python-cased version
            if exc.lower() == "warm-up":
                return "Warm-up"
            if exc.lower() == "non-contact":
                return "Non-contact"

    # For all other strings, apply standard title case
    return s.title()

if __name__ == "__main__":
    app.run(debug=True)