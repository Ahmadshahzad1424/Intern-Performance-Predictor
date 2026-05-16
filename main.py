import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InternPerformancePredictor:
    """
    Enterprise-grade model for predicting intern performance.
    Uses XGBoost regression to predict a performance score (0-100)
    based on completion time, feedback, and attendance.
    """
    
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = ['task_completion_time', 'feedback_rating', 'attendance_rate']

    def generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generates realistic synthetic data for intern performance.
        """
        np.random.seed(42)
        
        # task_completion_time (hours): 10 to 100
        task_completion_time = np.random.uniform(10, 100, n_samples)
        
        # feedback_rating (1-5 scale)
        feedback_rating = np.random.uniform(1, 5, n_samples)
        
        # attendance_rate (0-1)
        attendance_rate = np.random.uniform(0.7, 1.0, n_samples)
        
        # Performance Score (0-100) - Target
        # Lower completion time + higher feedback + higher attendance = higher score
        base_score = (100 - task_completion_time) * 0.4 + \
                     (feedback_rating * 15) + \
                     (attendance_rate * 20)
        
        # Add some noise
        noise = np.random.normal(0, 5, n_samples)
        performance_score = np.clip(base_score + noise, 0, 100)
        
        df = pd.DataFrame({
            'task_completion_time': task_completion_time,
            'feedback_rating': feedback_rating,
            'attendance_rate': attendance_rate,
            'performance_score': performance_score
        })
        
        logger.info(f"Generated synthetic dataset with {n_samples} samples.")
        return df

    def train(self, data: pd.DataFrame):
        """
        Trains the XGBoost model with feature scaling.
        """
        X = data[self.feature_names]
        y = data['performance_score']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("Training Intern Performance Model (XGBoost)...")
        self.model.fit(
            X_train_scaled, 
            y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Evaluation
        predictions = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        logger.info(f"Model Training Complete. MSE: {mse:.4f}, R2: {r2:.4f}")
        return X_test, y_test, predictions

    def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """
        Predicts performance for new interns.
        """
        scaled_data = self.scaler.transform(input_data[self.feature_names])
        return self.model.predict(scaled_data)

    def plot_feature_importance(self):
        """
        Visualizes which factors contribute most to intern success.
        """
        importance = self.model.feature_importances_
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance, y=self.feature_names, palette='viridis')
        plt.title('Key Drivers of Intern Performance')
        plt.xlabel('Importance Score')
        plt.show()

if __name__ == "__main__":
    predictor = InternPerformancePredictor()
    data = predictor.generate_synthetic_data()
    X_test, y_test, preds = predictor.train(data)
    
    # Showcase results
    print("\n--- Intern Performance Prediction Sample ---")
    sample_results = X_test.copy()
    sample_results['Actual_Score'] = y_test
    sample_results['Predicted_Score'] = preds
    print(sample_results.head())
    
    # Identify interns likely to struggle (Score < 50)
    struggling = sample_results[sample_results['Predicted_Score'] < 50]
    print(f"\nPotential struggling interns identified: {len(struggling)}")
