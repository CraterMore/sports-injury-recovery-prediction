from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/", methods=["GET"])
def ui():
    return render_template("application.html")

@app.route("/predict", methods=["POST"])
def predict():
    # get inputs from form
    age = request.form.get("age")
    gender = request.form.get("gender")
    minutes = request.form.get("minutes")
    injury = request.form.get("injury")
    body_part = request.form.get("body-part")
    severity = request.form.get("severity")
    cause = request.form.get("cause")
    recurrent = request.form.get("recurrent")
    sport = request.form.get("sport")
    competition = request.form.get("competition")
    region = request.form.get("region")
    event = request.form.get("event")
    surface = request.form.get("surface")

    # run model
    prediction = "predicted recovery: 14 days" #CHANGE

    return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)