"""
Test script for YOUR actual athlete injury data
"""

from regression import InjuryRecoveryPredictor

def test_with_real_data():
    """
    Test the model with your actual injury dataset
    """
    print("\n" + "="*60)
    print("ATHLETE INJURY RECOVERY PREDICTOR - REAL DATA")
    print("="*60)
    
    # Initialize predictor
    print("\n[1/7] Initializing predictor...")
    predictor = InjuryRecoveryPredictor()
    
    # Load YOUR data
    print("\n[2/7] Loading your data...")
    df = predictor.load_data('FinalInjuryData.csv')  # Replace with your actual filename
    if df is None:
        return
    
    # Preprocess - NOTE: Using 'DaysToRecovery' as target
    print("\n[4/7] Preprocessing data...")
    X, y = predictor.preprocess_data(df, target_column='DaysToRecovery')
    
    print("\nYour features:")
    for i, feature in enumerate(predictor.feature_columns, 1):
        print(f"  {i}. {feature}")
    
    # Split data
    print("\n[5/7] Splitting data...")
    X_train, X_test, y_train, y_test = predictor.split_data(X, y, test_size=0.2)
    
    # Train model
    print("\n[6/7] Training model...")
    results = predictor.train_model(
        X_train, y_train, X_test, y_test,
        learning_rate=0.01,  # Good starting point
        max_iter=1000
    )
    
    # Show results
    print("\n[7/7] Results...")
    predictor.show_results()
    
    # Feature coefficients
    print("\nAnalyzing which factors most affect recovery time...")
    coefficients = predictor.get_feature_importance()
    
    print("\n" + "="*60)
    print("TOP 5 FACTORS THAT INCREASE RECOVERY TIME:")
    print("="*60)
    top_positive = coefficients.nlargest(5, 'Coefficient')
    for idx, row in top_positive.iterrows():
        print(f"  • {row['Feature']}: +{row['Coefficient']:.2f} days")
    
    print("\n" + "="*60)
    print("TOP 5 FACTORS THAT DECREASE RECOVERY TIME:")
    print("="*60)
    top_negative = coefficients.nsmallest(5, 'Coefficient')
    for idx, row in top_negative.iterrows():
        print(f"  • {row['Feature']}: {row['Coefficient']:.2f} days")
    
    # Test predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    # Example 1: Based on your first data row
    test_case_1 = {
        'Age': 39,
        'Gender': 'Female',
        'MinutesPlayedLast30Days': 383,
        'InjuryType': 'Ligament Sprain',
        'BodyPart': 'Calf',
        'Severity': 'Minor',
        'InjuryCause': 'Contact',
        'IsRecurrentInjury': 'No',
        'Sport': 'Football',
        'CompetitionLevel': 'Professional',
        'Region': 'Northern Europe',
        'EventType': 'Warm-up',
        'SurfaceType': 'Grass',
        'TreatmentMethod': 'Massage',
        'Outcome': 'Fully Recovered',
        'CostOfTreatmentEuros': 89.03
    }
    
    print("\nTest Case 1: Professional female footballer, minor calf sprain")
    prediction_1 = predictor.predict(test_case_1)
    print(f"  Predicted: {prediction_1:.1f} days")
    print(f"  Actual: 12 days (from your data)")
    print(f"  Difference: {abs(prediction_1 - 12):.1f} days")
    
    # Example 2: Severe case
    test_case_2 = {
        'Age': 27,
        'Gender': 'Female',
        'MinutesPlayedLast30Days': 787,
        'InjuryType': 'Fracture',
        'BodyPart': 'Knee',
        'Severity': 'Moderate',
        'InjuryCause': 'Contact',
        'IsRecurrentInjury': 'Yes',
        'Sport': 'Football',
        'CompetitionLevel': 'Professional',
        'Region': 'Central Europe',
        'EventType': 'Training',
        'SurfaceType': 'Indoor Court',
        'TreatmentMethod': 'Physiotherapy',
        'Outcome': 'Retired',
        'CostOfTreatmentEuros': 1556.86
    }
    
    print("\nTest Case 2: Professional footballer, knee fracture (recurrent)")
    prediction_2 = predictor.predict(test_case_2)
    print(f"  Predicted: {prediction_2:.1f} days")
    print(f"  Actual: 44 days (from your data)")
    print(f"  Difference: {abs(prediction_2 - 44):.1f} days")
    
    # Save model
    print("\n" + "="*60)
    print("Saving model...")
    predictor.save_model('injury_recovery_model.pkl')
    
    print("\n" + "="*60)
    print("✓ ANALYSIS COMPLETE!")
    print("="*60)
    
    # Model interpretation
    print("\n📊 INSIGHTS FROM YOUR DATA:")
    print(f"  • Average recovery time: {y.mean():.1f} days")
    print(f"  • Range: {y.min():.0f} - {y.max():.0f} days")
    print(f"  • Model MAE: {results['MAE']:.2f} days")
    print(f"  • Model R²: {results['R2']:.3f}")
    
    if results['R2'] > 0.7:
        print("  • ✓ Good model fit! The features explain recovery time well.")
    elif results['R2'] > 0.4:
        print("  • ⚠ Moderate fit. Consider feature engineering or more data.")
    else:
        print("  • ⚠ Weak fit. May need different features or model type.")
    
    return predictor


def make_custom_prediction(predictor):
    """
    Helper function to make a custom prediction
    """
    print("\n" + "="*60)
    print("MAKE YOUR OWN PREDICTION")
    print("="*60)
    
    # Customize this with any values from your dataset
    custom_case = {
        'Age': 25,
        'Gender': 'Male',
        'MinutesPlayedLast30Days': 1200,
        'InjuryType': 'Muscle Strain',
        'BodyPart': 'Hamstring',
        'Severity': 'Moderate',
        'InjuryCause': 'Overuse',
        'IsRecurrentInjury': 'No',
        'Sport': 'Football',
        'CompetitionLevel': 'Professional',
        'Region': 'Western Europe',
        'EventType': 'Competition',
        'SurfaceType': 'Grass',
        'TreatmentMethod': 'Physiotherapy',
        'Outcome': 'Fully Recovered',
        'CostOfTreatmentEuros': 500.0
    }
    
    prediction = predictor.predict(custom_case)
    print(f"\nPredicted recovery time: {prediction:.1f} days")
    print("\nModify the custom_case dictionary above to test different scenarios!")


if __name__ == "__main__":
    # Run the test
    trained_predictor = test_with_real_data()
    
    # Make a custom prediction
    if trained_predictor:
        make_custom_prediction(trained_predictor)