# Sports Injury Recovery Prediction Tool

**CS 4243 Machine Learning Final Project**

Joshua Bearfield, Femke Jansen, Carter Moore, Fiona Pendergast, Katherine Tse

## Project Summary
A machine learning project that predicts optimal treatment methods and recovery timelines for various sports injuries. This tool combines two regression models to provide recovery insights for athletes and medical professionals.

![alt text](/static/images/input_form.png)
![alt text](/static/images/results_UI.png)

## Models
### Logistic Regression
**Purpose:** Predict the probability of full recovery for different treatment methods

**Implementation:** 
* Binomial logistic regression model that evaluates treatment effectiveness
* Iterates through available treatment options to identify those with highest probability of full recovery
* Replaced initial decision tree approach to avoid bias toward most common treatments


### Linear Regression
**Purpose:** Estimate recovery duration in days for given recovery method

**Implementation:** 
* Stochastic Gradient Descent (SGD) via Sklearn's SGDRegressor
* Standardized numerical features using StandardScaler
* Categorical features encoded with LabelEncoder

## Dataset
The dataset originated from a multi-tab spreadsheet containing sports injury records across multiple variables. Using pandas, we performed data consolidation by joining tables on primary keys and selecting relevant features for our predictive models.

![alt text](/static/images/data.png)

**Features Selected:**
| Category| Features |
|---|---|
|Athlete-Specific| Age, Gender, Minutes played in last 30 days|
|Injury-Related| Injury type, Body part affected, Severity (Minor/Moderate/Severe), Cause, Recurrence status|
|Event-Level| Sport, Competition level, Region, Event type, Surface type|
|Target Variables| Treatment method, Days to recover, Cost of recovery| 

### Dataset Characteristics

**Total Records:** ~15,000 samples

**Severity Distribution:**
* Minor: 50%
* Moderate: 35%
* Severe: 15%

**Distribution:** Most features (surface type, sport, age, treatment method, injury cause, event type) are evenly distributed

**Sport-Specific Patterns:** Certain injuries are exclusive to specific sports (e.g., nose, jaw, face injuries predominantly in boxing)

![alt text](/static/images/data_breakdown.png)

## Technologies Used
* Python
* Scikit-learn
    * Logistic Regression
    * SGDRegressor
    * GridSearchCV
    * StandardScalar
    * LabelEncoder 

## Results - Model Metrics and Performance

### Treatment Method Model (Logistic Regression)
- **Accuracy:** 78%
- Predicts the probability of full recovery for different treatment methods

### Recovery Time Model (Linear Regression with SGD)
- **Mean Absolute Error (MAE):** 32.97 days
- **Root Mean Squared Error (RMSE):** [Your RMSE value]
- **R² Score:** 0.640
- **Interpretation:** 
  - The model explains 64% of the variance in recovery times
  - Average prediction error is approximately 33 days
  - Validated using 5-fold cross-validation

### Key Hyperparameters
- **Learning Rate:** 0.01
- **Max Iterations:** 1,000
- **Regularization:** L2 (Ridge) with α = 0.0001
- **Early Stopping Tolerance:** 10⁻³

### Model Features
- **Preprocessing:** StandardScaler normalization + Label Encoding for categorical variables
- **Optimization:** Stochastic Gradient Descent (SGD)
- **Interpretability:** Coefficient analysis reveals feature impact on recovery time

## Installation & Usage

```bash
# Clone repository
git clone https://github.com/CraterMore/sports-injury-recovery-prediction.git

# Install dependencies
pip install -r requirements.txt

# Run the program 
python ui.py
# or Alternative top 3 option
python top_3_ui.py

# Open the localhost in your browser to use the UI
# http://127.0.0.1:5000/
```

## Alternate Results Display

By altering our output to display the probability of a recovery not ending in retirement, we were able to display a much higher chance of a successful recovery. We determined the factors that might be the most useful for a user of this project would be more information across multiple recovery methods so we also implemented a "Top 3" recovery results page. 

![alt text](/static/images/top3_results_UI.png)

## Contributors
| Name | Major | Grad Year |
| --- | --- | --- |
| Joshua Bearfield | Computer Science | 2027 |
| Femke Jansen | Computer Science | 2027 |
| Carter Moore | Computer Science | 2026 |
| Fiona Pendergast | Computer Science + Robotics Engineering | 2026 |
| Katherine Tse | Computer Science + Interactive Media and Game Development | 2026 |

## Course Information
Worcester Polytechnic Institute

CS 4243 Machine Learning A-Term 2025

Professor Kyumin Lee

## Acknowledgements
**Dataset:** Our dataset was created by FP20 for a data analysis challenge linked below.
* [Dataset Spreadsheet Link](https://docs.google.com/spreadsheets/d/1e0OpTErDDSlV1JxID5GKURyCzn71U0mm/edit?gid=1125320447#gid=112532044)
* [Data Analysis Challenge Information Link](https://docs.google.com/document/d/1DAs8Ayw6lrxyyJC42ry_F-mKwwsDg9vL/edit)