import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# opens csv in read only
file = open("FinalInjuryData.csv", "r")

# reads the data into a pandas dataframe object, replacing entries of outcome table with 1 for fully recovered or 0 otherwise
data = pd.read_csv(file, sep=",", converters={'Outcome': lambda x: int(x == 'Fully Recovered')})

# splits the data into the instances and labels
y = data.Outcome
X = data.drop('Outcome', axis=1).drop('DaysToRecovery', axis=1).drop('CostOfTreatmentEuros', axis=1)

# extracts categorical features from X and turns them into numerical features using pandas factorize
cat_columns = X.select_dtypes(object).columns
X[cat_columns] = X[cat_columns].apply(lambda x: pd.factorize(x)[0])

# splits data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardizes the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# trains logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# evaluates accuracy of model using testing set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# print more detailed classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, zero_division=0))