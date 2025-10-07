import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

##Reference for code: https://www.geeksforgeeks.org/machine-learning/how-to-optimize-logistic-regression-performance/

#import data
x = []
y = []

## split data into training and testing 
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)

#run model here
log_model = 'add model'

###Tuning Hyperparameters###
#grid for tuning (can def be changed)
param_grid = [
    {
        'pentaly': ['l1', 'l2'], #what regularization methods do you want to use
        'C': np.logspace(-4,4,5), #inverse of regularization strength, right now it is generated 20 values between 10^-4 to 10^4 (small C -> stronger regularization, large C -> weaker regularization[model fits closer to training data])
        'solver': ['liblinear', 'saga'], #optimization algorithms 'lbfgs': Good default, handles L2 penalty, 'newton-cg': Works with L2, better for large datasets, 'sag': Fast for large datasets with L2, 'saga': Supports L1, L2, and elasticnet
        'max_iter': [1000] #max number of iterations for solver
    }
    ]

## run grid search to find optimal set of hyperparameters
grid_results = GridSearchCV(log_model, param_grid = param_grid, cv = 3, verbose = True, n_jobs = -1)

## fit best hyperparameter log regression on training data 
best = grid_results.fit(x, y)

## how to check results to see if it worked 
# best.best_estimator_
# best.score(x,y) meow meow