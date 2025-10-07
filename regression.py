import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

class InjuryRecoveryPredictor:
    '''
    A class to predict injury recovery time using a regression model.
    '''

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.results = {}

    def load_data(self, filepath):
        '''
        Load dataset from a CSV file.
        Input:
            filepath: str, path to the CSV file.
        Output:
            data: pandas DataFrame, loaded dataset.
        '''
        self.data = pd.read_csv(filepath)
        print(f"Data loaded from {filepath}. Shape: {self.data.shape}")
        return self.data
    
    # def explore_data(self, df):
    #     '''
    #     Print basic information about the dataset.
    #     Input:
    #         df (pd.DataFrame): Input dataframe
    #     '''
    #     print("\n" + "="*60)
    #     print("DATA EXPLORATION")
    #     print("="*60)
    #     print(f"\nDataset shape: {df.shape}")
    #     print("\nFirst few rows:")
    #     print(df.head())
    #     print("\nData types:")
    #     print(df.dtypes)
    #     print("\nMissing values:")
    #     print(df.isnull().sum())
    #     print("\nBasic statistics:")
    #     print(df.describe())

    def preprocess_data(self, df, target_column='recovery_time'):
        '''
        Preprocess the data: handle missing values and encode categorical variables.
        Input:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the target variable column
        Output:
            tuple: (X, y) - features and target variable
        '''
        df = df.copy()
        
        # Handle missing values for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Handle missing values for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Encode categorical variables
        for col in categorical_cols:
            if col != target_column:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        self.feature_columns = X.columns.tolist()
        
        print(f"\nPreprocessing complete!")
        print(f"Features: {self.feature_columns}")
        print(f"Target: {target_column}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        '''
        Split data into training and testing sets.
        Input:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility  
        Output:
            tuple: (X_train, X_test, y_train, y_test)
        '''
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nData split complete!")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        '''
        Scale features using StandardScaler.
        Important to scale features for gradient descent to work
        Input:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            
        Output:
            tuple: (X_train_scaled, X_test_scaled)
        '''
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\nFeatures scaled for gradient descent optimization")
        
        return X_train_scaled, X_test_scaled
    

    
    def train_model(self, X_train, y_train, X_test, y_test, 
                    learning_rate=0.01, max_iter=1000, penalty='l2', alpha=0.0001):
        '''
        Train a Linear Regression model using Stochastic Gradient Descent (SGD).
        Input:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            learning_rate: Learning rate for gradient descent (default: 0.01)
            max_iter: Maximum number of iterations (default: 1000)
            penalty: Regularization type - 'l2' (ridge), 'l1' (lasso), or None (default: 'l2')
            alpha: Regularization strength (default: 0.0001) 
        Output:
            dict: Results with performance metrics
        '''
        print("\n" + "="*60)
        print("MODEL TRAINING - Linear Regression with Gradient Descent")
        print("="*60)
        
        # Scale features (REQUIRED for gradient descent!)
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Initialize SGD Regressor (Linear Regression with Gradient Descent)
        self.model = SGDRegressor(
            loss='squared_error',  # This makes it linear regression
            penalty=penalty,
            alpha=alpha,
            learning_rate='constant',
            eta0=learning_rate,
            max_iter=max_iter,
            random_state=42,
            tol=1e-3
        )
        
        print(f"\nTraining with:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Max iterations: {max_iter}")
        print(f"  Regularization: {penalty}")
        print(f"  Alpha: {alpha}")
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, 
                                   cv=5, scoring='neg_mean_absolute_error')
        
        self.results = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'CV_MAE': -cv_scores.mean(),
            'iterations': self.model.n_iter_,
            'converged': self.model.n_iter_ < max_iter
        }
        
        print(f"\n  Iterations to converge: {self.model.n_iter_}")
        print(f"  MAE: {mae:.2f} days")
        print(f"  RMSE: {rmse:.2f} days")
        print(f"  R²: {r2:.3f}")
        print(f"  CV MAE: {-cv_scores.mean():.2f} days")
        
        if not self.results['converged']:
            print(f"\n  WARNING: Model did not converge! Consider increasing max_iter.")
        
        return self.results
    
    def show_results(self):
        '''
        Display the model performance results.
        '''
        print("\n" + "="*60)
        print("MODEL RESULTS")
        print("="*60)
        for metric, value in self.results.items():
            print(f"{metric}: {value:.3f}")
    
    def get_feature_importance(self):
        '''
        Get feature coefficients from the linear model.
        For linear regression, coefficients show the impact of each feature.
        Output:
            pd.DataFrame: Feature coefficients (positive = increases recovery time)
        '''
        if self.model is None:
            print("Model has not been trained yet.")
            return None
            
        coefficients_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Coefficient': self.model.coef_
        }).sort_values('Coefficient', ascending=False, key=abs)
        
        print("\n" + "="*60)
        print("FEATURE COEFFICIENTS")
        print("="*60)
        print("Positive = increases recovery time")
        print("Negative = decreases recovery time")
        print("Larger magnitude = stronger effect\n")
        print(coefficients_df)
        
        return coefficients_df
    
    def predict(self, input_data):
        '''
        Make predictions using the trained model.
        Input:
            input_data (dict or pd.DataFrame): Input features for prediction
        Output:
            float or np.array: Predicted recovery time(s)
        '''
        if self.model is None:
            raise ValueError("No model has been trained yet. Call train_model() first.")
        
        # Convert dict to DataFrame if necessary
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Ensure columns match training data
        for col in self.feature_columns:
            if col not in input_df.columns:
                raise ValueError(f"Missing feature: {col}")
        
        input_df = input_df[self.feature_columns]
        
        # Apply label encoding to categorical features
        for col, encoder in self.label_encoders.items():
            if col in input_df.columns:
                input_df[col] = encoder.transform(input_df[col].astype(str))
        
        # Scale features (required for gradient descent model!)
        input_scaled = self.scaler.transform(input_df)
        
        # Make prediction
        prediction = self.model.predict(input_scaled)
        
        return prediction[0] if len(prediction) == 1 else prediction
    
    def save_model(self, filepath='injury_recovery_model.pkl'):
        '''
        Save the trained model and preprocessing objects.
        Input:
            filepath (str): Path to save the model
        '''
        if self.model is None:
            raise ValueError("No model has been trained yet.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath='injury_recovery_model.pkl'):
        '''
        Load a previously saved model.
        Input:
            filepath (str): Path to the saved model
        '''
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        
        print(f"Model loaded from {filepath}")
        print(f"Model type: Linear Regression (Gradient Descent)")


def main():
    '''
    Example usage of the InjuryRecoveryPredictor class.
    '''
    # Initialize predictor
    predictor = InjuryRecoveryPredictor()
    
    # Load data
    df = predictor.load_data('your_data.csv')
    if df is None:
        return
    
    # Explore data
    predictor.explore_data(df)
    
    # Preprocess data
    X, y = predictor.preprocess_data(df, target_column='recovery_time')
    
    # Split data
    X_train, X_test, y_train, y_test = predictor.split_data(X, y)
    
    # Train model
    results = predictor.train_model(X_train, y_train, X_test, y_test)
    
    # Show results
    predictor.show_results()
    
    # Get feature importance
    predictor.get_feature_importance()
    
    # Save model
    predictor.save_model('injury_recovery_model.pkl')
    
    # Example prediction
    # sample_input = {
    #     'age': 25,
    #     'injury_type': 'sprain',
    #     'severity': 2,
    #     # ... other features
    # }
    # prediction = predictor.predict(sample_input)
    # print(f"\nPredicted recovery time: {prediction:.2f} days")


if __name__ == "__main__":
    main()
    
