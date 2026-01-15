"""
Machine Learning Module for Disease Spread Prediction
Implements basic ML algorithms to predict future infection trends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path

class DiseasePredictor:
    """Machine learning predictor for disease spread"""
    
    def __init__(self):
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        
    def prepare_data(self, filepath: str = 'data/processed/ml_training_data.csv'):
        """Load and prepare data for training"""
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} samples")
        
        # Define features and target
        feature_cols = [col for col in df.columns if col not in ['day', 'infected', 'susceptible', 'recovered', 'new_cases']]
        target_col = 'infected'
        
        if not feature_cols:
            print("No feature columns found. Using basic features...")
            # Create simple features from available data
            df['day_squared'] = df['day'] ** 2
            df['day_cubed'] = df['day'] ** 3
            feature_cols = ['day', 'day_squared', 'day_cubed']
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        print(f"Features: {feature_cols}")
        print(f"Target: {target_col}")
        
        return X, y, feature_cols
    
    def train_models(self, X, y, test_size=0.2):
        """Train multiple models and compare performance"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        print("\nTraining models...")
        print("-" * 60)
        
        for name, model in self.models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_r2 = r2_score(y_test, y_pred_test)
            
            results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'predictions': y_pred_test,
                'actual': y_test
            }
            
            print(f"{name}:")
            print(f"  Train RMSE: {train_rmse:.2f}")
            print(f"  Test RMSE:  {test_rmse:.2f}")
            print(f"  Test MAE:   {test_mae:.2f}")
            print(f"  Test R²:    {test_r2:.4f}")
            print()
        
        # Select best model based on test RMSE
        self.best_model_name = min(results.keys(), key=lambda k: results[k]['test_rmse'])
        self.best_model = results[self.best_model_name]['model']
        
        print(f"Best model: {self.best_model_name}")
        print("-" * 60)
        
        return results, X_test_scaled, y_test
    
    def predict_future(self, last_day_features: np.ndarray, days_ahead: int = 30):
        """Predict infections for future days"""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        predictions = []
        current_features = last_day_features.copy()
        
        for _ in range(days_ahead):
            # Scale features
            current_features_scaled = self.scaler.transform(current_features.reshape(1, -1))
            
            # Predict
            pred = self.best_model.predict(current_features_scaled)[0]
            predictions.append(max(0, pred))  # Ensure non-negative
            
            # Update features for next prediction (simple approach)
            # In reality, you'd update based on predicted values
            current_features = current_features  # Simplified
        
        return np.array(predictions)
    
    def visualize_results(self, results, X_test, y_test):
        """Visualize model performance"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Predictions vs Actual for all models
        for i, (name, result) in enumerate(results.items()):
            ax = axes[i // 2, i % 2]
            ax.scatter(result['actual'], result['predictions'], alpha=0.5)
            ax.plot([result['actual'].min(), result['actual'].max()],
                   [result['actual'].min(), result['actual'].max()],
                   'r--', lw=2)
            ax.set_xlabel('Actual Infections')
            ax.set_ylabel('Predicted Infections')
            ax.set_title(f'{name}\nR² = {result["test_r2"]:.4f}')
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot if odd number of models
        if len(results) < 4:
            fig.delaxes(axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 2: Time series comparison for best model
        fig, ax = plt.subplots(figsize=(12, 6))
        
        best_result = results[self.best_model_name]
        indices = np.arange(len(best_result['actual']))
        
        ax.plot(indices, best_result['actual'], 'b-', label='Actual', linewidth=2)
        ax.plot(indices, best_result['predictions'], 'r--', label='Predicted', linewidth=2)
        ax.fill_between(indices, best_result['actual'], best_result['predictions'], alpha=0.3)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Number of Infections')
        ax.set_title(f'Best Model ({self.best_model_name}) - Predictions vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/best_model_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filename: str = 'results/disease_predictor.pkl'):
        """Save trained model"""
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'model_name': self.best_model_name
        }
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename: str = 'results/disease_predictor.pkl'):
        """Load trained model"""
        model_data = joblib.load(filename)
        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        self.best_model_name = model_data['model_name']
        print(f"Loaded model: {self.best_model_name}")

def evaluate_on_simulation_data(simulation_csv: str = 'simulation_results.csv'):
    """Train and evaluate models on simulation output"""
    print("=" * 60)
    print("Machine Learning Prediction on Simulation Data")
    print("=" * 60)
    
    # Load simulation data
    df = pd.read_csv(simulation_csv)
    print(f"\nLoaded simulation data: {len(df)} days")
    
    # Create features
    df['day_squared'] = df['day'] ** 2
    df['susceptible_ratio'] = df['susceptible'] / (df['susceptible'] + df['infected'] + df['recovered'])
    df['infected_lag_1'] = df['infected'].shift(1)
    df['infected_lag_7'] = df['infected'].shift(7)
    df['new_infections_lag_1'] = df['new_infections'].shift(1)
    
    # Drop NaN rows
    df = df.dropna()
    
    # Prepare features and target
    feature_cols = ['day', 'day_squared', 'susceptible', 'recovered', 
                   'infected_lag_1', 'infected_lag_7', 'new_infections_lag_1']
    X = df[feature_cols].values
    y = df['infected'].values
    
    # Train models
    predictor = DiseasePredictor()
    results, X_test, y_test = predictor.train_models(X, y, test_size=0.2)
    
    # Visualize results
    predictor.visualize_results(results, X_test, y_test)
    
    # Save model
    Path('results').mkdir(exist_ok=True)
    predictor.save_model()
    
    return predictor, results

def main():
    """Main ML pipeline"""
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Try to load processed data
    try:
        predictor = DiseasePredictor()
        X, y, feature_cols = predictor.prepare_data()
        
        # Train models
        results, X_test, y_test = predictor.train_models(X, y)
        
        # Visualize
        predictor.visualize_results(results, X_test, y_test)
        
        # Save model
        predictor.save_model()
        
    except FileNotFoundError:
        print("Processed data not found. Please run data_preparation.py first")
        print("Or run the simulation to generate simulation_results.csv")

if __name__ == "__main__":
    main()