import pandas as pd
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for

from regression import InjuryRecoveryPredictor

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flash messages

# load trained model
predictor = InjuryRecoveryPredictor()
predictor.load_model("injury_recovery_model.pkl")

# Print the class order for logistic regression model
# print("\n" + "="*60)
# print("LOGISTIC REGRESSION MODEL - CLASS INFORMATION")
# print("="*60)
# print(f"Classes: {predictor.model_clf.classes_}")
# print(f"Number of classes: {len(predictor.model_clf.classes_)}")
# print("\nClass encoding:")
# for i, class_name in enumerate(predictor.model_clf.classes_):
#     print(f"  Index [{i}] = '{class_name}'")
# print("="*60 + "\n")

# form page
@app.route("/", methods=["GET"])
def ui():
    return render_template("application.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Define required fields
    required_fields = {
        "age": "Age",
        "gender": "Gender",
        "minutes": "Minutes Played",
        "injury": "Injury Type",
        "body-part": "Body Part",
        "severity": "Severity",
        "cause": "Injury Cause",
        "recurrent": "Recurrent Injury",
        "sport": "Sport",
        "competition": "Competition Level",
        "region": "Region",
        "event": "Event Type",
        "surface": "Surface Type"
    }
    
    # Check for missing fields
    missing_fields = []
    for field_name, display_name in required_fields.items():
        value = request.form.get(field_name)
        if not value or value.strip() == "":
            missing_fields.append(display_name)
    
    # If any fields are missing, return error
    if missing_fields:
        error_message = f"Please fill in the following required fields: {', '.join(missing_fields)}"
        flash(error_message, 'error')
        return redirect(url_for('ui'))
    
    # all possible treatment methods
    all_treatment_methods = ['Massage', 'Surgery', 'Physiotherapy', 'Rest', 'Ice/Heat Therapy', 'Medication']

    # get fixed inputs from form
    try:
        age = float(request.form.get("age"))
        minutes = float(request.form.get("minutes"))
    except (ValueError, TypeError):
        flash("Age and Minutes Played must be valid numbers", 'error')
        return redirect(url_for('ui'))
    
    gender = to_title_case(request.form.get("gender"))
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

    # Store results for all treatments
    treatment_results = []

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

        # linear regression prediction -> recovery time (days) (absolute value to avoid negative days)
        recovery_time = abs(round(predictor.model_reg.predict(input_scaled)[0]))

        # Logistic regression prediction -> probability of recovery (not retired)
        if hasattr(predictor.model_clf, "predict_proba"):
            probabilities = predictor.model_clf.predict_proba(input_scaled)[0]
            # Assuming classes are ordered: [0: Fully Recovered, 1: Recovered with Limitations, 2: Retired]
            retired_prob = probabilities[2]
            recovery_prob = 1 - retired_prob  # P(any recovery) = 1 - P(retired)
        else:
            recovery_prob = predictor.model_clf.predict(input_scaled)[0]

        print(f"Treatment: {treatment}")
        print(f"Retired probability: {retired_prob:.3f}")
        print(f"Recovery probability: {recovery_prob:.3f}")
        
        # Store all results
        treatment_results.append({
            'treatment': treatment,
            'recovery_time': recovery_time,
            'recovery_prob': round(recovery_prob, 2)
        })

    # Sort by recovery probability (highest first)
    treatment_results.sort(key=lambda x: x['recovery_prob'], reverse=True)
    
    # Get top 3 treatments
    top_3_treatments = treatment_results[:3]
    
    print("\nTop 3 Treatments:")
    for i, result in enumerate(top_3_treatments, 1):
        print(f"{i}. {result['treatment']}: {result['recovery_prob']*100}% probability, {result['recovery_time']} days")

    return render_template(
        "top3_results.html",
        top_treatments=top_3_treatments
    )


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