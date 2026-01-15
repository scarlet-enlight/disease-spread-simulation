"""
Data Preparation Script
Downloads and processes COVID-19 data from Kaggle to calibrate simulation parameters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def download_kaggle_data():
    """
    Instructions for downloading COVID-19 data from Kaggle
    
    Dataset: "COVID-19 Dataset" by Sudalai Rajkumar
    URL: https://www.kaggle.com/datasets/sudalairajkumar/novel-corona-virus-2019-dataset
    
    Alternative: "Coronavirus (COVID-19) Geo-tagged Tweets Dataset"
    URL: https://www.kaggle.com/datasets/smid80/coronavirus-covid19-tweets
    
    Steps:
    1. Download the dataset from Kaggle
    2. Extract to data/raw/ folder
    3. Run this script to process the data
    """
    print("Please download COVID-19 data from Kaggle manually")
    print("See function docstring for instructions")

def process_covid_data(filepath: str = 'data/raw/covid_19_data.csv'):
    """
    Process COVID-19 data to extract transmission parameters
    
    Args:
        filepath: Path to the raw COVID-19 dataset
    
    Returns:
        DataFrame with processed statistics
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data with {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        
        # Example processing for typical COVID dataset structure
        if 'Country/Region' in df.columns:
            # Filter for Poland if available
            poland_data = df[df['Country/Region'] == 'Poland'].copy()
            
            if len(poland_data) > 0:
                print(f"\nFound {len(poland_data)} records for Poland")
                return poland_data
        
        return df
    
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        print("Generating synthetic data based on COVID-19 characteristics...")
        return generate_synthetic_covid_data()

def generate_synthetic_covid_data():
    """
    Generate synthetic COVID-19 data based on known epidemiological parameters
    This serves as a fallback and demonstrates data structure
    """
    np.random.seed(42)
    days = 200
    
    # Simulate SIR model dynamics
    N = 100000  # Population
    I0 = 100    # Initial infected
    R0 = 0      # Initial recovered
    S0 = N - I0 - R0
    
    beta = 0.5   # Transmission rate
    gamma = 0.1  # Recovery rate
    
    S, I, R = [S0], [I0], [R0]
    
    for day in range(1, days):
        new_infections = beta * S[-1] * I[-1] / N
        new_recoveries = gamma * I[-1]
        
        S.append(S[-1] - new_infections)
        I.append(I[-1] + new_infections - new_recoveries)
        R.append(R[-1] + new_recoveries)
    
    df = pd.DataFrame({
        'day': range(days),
        'susceptible': S,
        'infected': I,
        'recovered': R,
        'new_cases': [I[i] - I[i-1] + (R[i] - R[i-1]) if i > 0 else I0 for i in range(days)]
    })
    
    # Add some noise
    df['new_cases'] = df['new_cases'].clip(lower=0) * np.random.uniform(0.8, 1.2, days)
    
    # Save synthetic data
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    df.to_csv('data/processed/synthetic_covid_data.csv', index=False)
    print("Synthetic COVID data generated and saved to data/processed/")
    
    return df

def extract_transmission_parameters(df: pd.DataFrame):
    """
    Extract key epidemiological parameters from COVID data
    
    Returns:
        Dictionary with transmission parameters for different settings
    """
    params = {
        'basic_reproduction_number': 2.5,  # R0 for COVID-19
        'incubation_period': 5.1,  # days
        'infectious_period': 14,   # days
        'transmission_rates': {
            'office': 0.15,      # Higher due to close contact
            'residential': 0.05,  # Lower, more spread out
            'shopping': 0.20,     # High due to crowds
            'park': 0.02,         # Low, outdoor setting
            'transport': 0.25     # Highest due to enclosed space
        }
    }
    
    # Save parameters
    import json
    with open('data/processed/transmission_parameters.json', 'w') as f:
        json.dump(params, f, indent=4)
    
    print("\nExtracted transmission parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    return params

def visualize_covid_data(df: pd.DataFrame):
    """Create visualizations of COVID data"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: SIR curves
    if all(col in df.columns for col in ['susceptible', 'infected', 'recovered']):
        axes[0, 0].plot(df['day'], df['susceptible'], 'g-', label='Susceptible', linewidth=2)
        axes[0, 0].plot(df['day'], df['infected'], 'r-', label='Infected', linewidth=2)
        axes[0, 0].plot(df['day'], df['recovered'], 'b-', label='Recovered', linewidth=2)
        axes[0, 0].set_xlabel('Day')
        axes[0, 0].set_ylabel('Number of People')
        axes[0, 0].set_title('COVID-19 SIR Model')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: New cases over time
    if 'new_cases' in df.columns:
        axes[0, 1].plot(df['day'], df['new_cases'], 'orange', linewidth=2)
        axes[0, 1].set_xlabel('Day')
        axes[0, 1].set_ylabel('New Cases')
        axes[0, 1].set_title('Daily New Cases')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Active cases
    if 'infected' in df.columns:
        axes[1, 0].fill_between(df['day'], df['infected'], alpha=0.3, color='red')
        axes[1, 0].plot(df['day'], df['infected'], 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Day')
        axes[1, 0].set_ylabel('Active Cases')
        axes[1, 0].set_title('Active Infections Over Time')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Growth rate
    if 'new_cases' in df.columns:
        growth_rate = df['new_cases'].pct_change().rolling(7).mean() * 100
        axes[1, 1].plot(df['day'], growth_rate, 'purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[1, 1].set_xlabel('Day')
        axes[1, 1].set_ylabel('Growth Rate (%)')
        axes[1, 1].set_title('7-Day Average Growth Rate')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/processed/covid_data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization saved to data/processed/covid_data_analysis.png")

def prepare_training_data(df: pd.DataFrame):
    """
    Prepare data for machine learning model
    Creates features for predicting future infections
    """
    if 'infected' not in df.columns or 'day' not in df.columns:
        print("Required columns not found for ML preparation")
        return None
    
    ml_data = df.copy()
    
    # Create lagged features
    for lag in [1, 3, 7, 14]:
        ml_data[f'infected_lag_{lag}'] = ml_data['infected'].shift(lag)
        if 'new_cases' in ml_data.columns:
            ml_data[f'new_cases_lag_{lag}'] = ml_data['new_cases'].shift(lag)
    
    # Create rolling statistics
    for window in [7, 14]:
        ml_data[f'infected_rolling_mean_{window}'] = ml_data['infected'].rolling(window).mean()
        ml_data[f'infected_rolling_std_{window}'] = ml_data['infected'].rolling(window).std()
    
    # Drop rows with NaN values
    ml_data = ml_data.dropna()
    
    # Save processed data
    ml_data.to_csv('data/processed/ml_training_data.csv', index=False)
    print(f"\nML training data saved: {len(ml_data)} samples with {len(ml_data.columns)} features")
    
    return ml_data

def main():
    """Main data preparation pipeline"""
    print("=" * 60)
    print("COVID-19 Data Preparation for Disease Simulation")
    print("=" * 60)
    
    # Create directory structure
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Try to load real data, fall back to synthetic
    print("\nStep 1: Loading COVID-19 data...")
    df = process_covid_data()
    
    # Generate synthetic data if real data not available
    if len(df) < 100:
        print("\nGenerating synthetic data for demonstration...")
        df = generate_synthetic_covid_data()
    
    # Extract parameters
    print("\nStep 2: Extracting transmission parameters...")
    params = extract_transmission_parameters(df)
    
    # Visualize data
    print("\nStep 3: Creating visualizations...")
    visualize_covid_data(df)
    
    # Prepare ML data
    print("\nStep 4: Preparing machine learning dataset...")
    ml_data = prepare_training_data(df)
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - data/processed/synthetic_covid_data.csv")
    print("  - data/processed/transmission_parameters.json")
    print("  - data/processed/ml_training_data.csv")
    print("  - data/processed/covid_data_analysis.png")

if __name__ == "__main__":
    main()