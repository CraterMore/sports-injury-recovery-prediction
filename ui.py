from flask import Flask, render_template, request, jsonify

from regression import InjuryRecoveryPredictor

app = Flask(__name__)

# load trained model
predictor = InjuryRecoveryPredictor()
predictor.load_model("real_injury_model.pkl")

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
        'Outcome': 'Fully Recovered',
        'TreatmentMethod': 'Rest', #default
        'CostOfTreatmentEuros': 1800.0 # default, average across test data
    }

    # linear regression prediction -> recovery time (days)
    recovery_time = round(predictor.predict(input_data))

    # logistic regression prediction -> probability of making full recovery based on treatment method
    treatment_method =

    return render_template("result.html", recovery_time=recovery_time, treatment_method=treatment_method)

if __name__ == "__main__":
    app.run(debug=True)