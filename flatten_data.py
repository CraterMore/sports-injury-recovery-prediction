import pandas as pd

# Load CSVs
injuries = pd.read_csv("factInjuries.csv")
players = pd.read_csv("dimPlayer.csv")
injury_types = pd.read_csv("dimInjuryType.csv")
locations = pd.read_csv("dimLocation.csv")
clubs = pd.read_csv("dimClub.csv")
events = pd.read_csv("dimEvent.csv")
treatments = pd.read_csv("dimTreatment.csv")

# merge fact table with dimension tables
merged = (
    injuries
    .merge(players[["PlayerDimKey", "Age", "Gender"]], on="PlayerDimKey", how="left")
    .merge(injury_types[["InjuryTypeDimKey", "InjuryType", "BodyPart", "Severity", "InjuryCause", "IsRecurrentInjury"]], on="InjuryTypeDimKey", how="left")
    .merge(locations[["LocationDimKey", "Region"]], on="LocationDimKey", how="left")
    .merge(clubs[["ClubDimKey", "Sport", "CompetitionLevel", "SurfaceType"]], on="ClubDimKey", how="left")
    .merge(events[["EventDimKey", "EventType"]], on="EventDimKey", how="left")
    .merge(treatments[["TreatmentDimKey", "TreatmentMethod", "Outcome"]], on="TreatmentDimKey", how="left")
)

# Select only columns we need
final = merged[
    [
        "Age",
        "Gender",
        "MinutesPlayedLast30Days",
        "InjuryType",
        "BodyPart",
        "Severity",
        "InjuryCause",
        "IsRecurrentInjury",
        "Sport",
        "CompetitionLevel",
        "Region",
        "EventType",
        "SurfaceType",
        "TreatmentMethod",
        "Outcome",
        "DaysToRecovery",
        "CostOfTreatmentEuros"
    ]
]

# save to csv
final.to_csv("FinalInjuryData.csv", index=False)

print(final.head())