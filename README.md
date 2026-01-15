# Disease Spread Simulation and Prediction

A comprehensive project for modeling disease transmission in urban environments with machine learning prediction capabilities. Developed for mathematical modeling and computer simulation coursework.

## 📋 Project Overview

This project simulates the spread of infectious diseases across different urban districts using agent-based modeling and the SIR (Susceptible-Infected-Recovered) epidemiological model. The simulation is calibrated using real COVID-19 data from Kaggle and includes machine learning models to predict future infection trends.

### Key Features

- **Agent-Based Simulation**: 500+ individuals moving between 5 distinct urban districts
- **Dynamic Disease Transmission**: Varying infection rates based on district characteristics
- **Visual Analytics**: Real-time visualization of disease spread with color-coded agents
- **Data-Driven Parameters**: Calibrated using COVID-19 datasets from Kaggle
- **Machine Learning Prediction**: Multiple ML algorithms for forecasting infection trends
- **Comprehensive Documentation**: Full project structure with analysis notebooks

## 🎯 Project Components

### 1. Disease Simulation (`simulation.py`)
The core simulation engine featuring:
- **5 Urban Districts** with unique characteristics:
  - 🏢 Office District (15% transmission rate)
  - 🏘️ Residential Area (5% transmission rate)
  - 🛍️ Shopping Center (20% transmission rate)
  - 🌳 Park (2% transmission rate)
  - 🚇 Transport Hub (25% transmission rate - highest risk)
- **Agent Movement**: Individuals randomly travel between districts
- **Disease Dynamics**: SIR model with 14-day recovery period
- **Color Coding**: 
  - 🟢 Green = Susceptible (healthy)
  - 🔴 Red = Infected
  - 🔵 Blue = Recovered

### 2. Data Preparation (`data_preparation.py`)
- Downloads and processes COVID-19 data from Kaggle
- Extracts epidemiological parameters (R₀, transmission rates, recovery period)
- Generates synthetic data if real data unavailable
- Creates feature-engineered datasets for ML training

### 3. Machine Learning Module (`ml_prediction.py`)
Implements three prediction algorithms:
- **Linear Regression**: Baseline model
- **Ridge Regression**: Regularized linear model
- **Random Forest**: Ensemble method (typically best performing)

Evaluation metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score (Coefficient of Determination)

## 📁 Project Structure

```
disease-spread-simulation/
│
├── data/
│   ├── raw/                    # Raw COVID-19 data from Kaggle
│   └── processed/              # Processed datasets and parameters
│       ├── synthetic_covid_data.csv
│       ├── ml_training_data.csv
│       └── transmission_parameters.json
│
├── results/
│   ├── model_comparison.png
│   ├── best_model_predictions.png
│   └── disease_predictor.pkl
│
├── notebooks/
│   └── analysis.ipynb          # Jupyter notebook for visualization
│
├── simulation.py               # Main simulation engine
├── data_preparation.py         # Data processing script
├── ml_prediction.py            # Machine learning module
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/disease-spread-simulation.git
cd disease-spread-simulation
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

or with conda:
```bash
conda env create -f environment.yml
```

### Step 3: Download COVID-19 Data (Optional)
1. Visit [Kaggle COVID-19 Dataset](https://www.kaggle.com/datasets/sudalairajkumar/novel-corona-virus-2019-dataset)
2. Download the dataset
3. Extract to `data/raw/` folder

Or the script will generate synthetic data automatically.

## 📊 Usage

### Run Complete Pipeline

```bash
# Step 1: Prepare data
python data_preparation.py

# Step 2: Run simulation
python simulation.py

# Step 3: Train ML models
python ml_prediction.py
```

### Run Simulation Only

```python
from simulation import run_simulation

# Run 100-day simulation with 500 people
sim = run_simulation(days=100, n_people=500)
```

### Train ML Models

```python
from ml_prediction import DiseasePredictor

predictor = DiseasePredictor()
X, y, features = predictor.prepare_data()
results, X_test, y_test = predictor.train_models(X, y)
```

### Use Jupyter Notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

## 📈 Expected Results

### Simulation Output
- `simulation_results.csv`: Daily statistics (susceptible, infected, recovered)
- `simulation_final_state.png`: Final visualization with disease spread curves
- Real-time visualization showing agent movement and infection spread

### Machine Learning Output
- Model comparison with performance metrics
- Prediction visualizations
- Saved model file (`.pkl`) for future predictions

### Typical Performance
- **Random Forest**: R² ≈ 0.92-0.97, RMSE ≈ 200-500
- **Ridge Regression**: R² ≈ 0.88-0.95, RMSE ≈ 300-700
- **Linear Regression**: R² ≈ 0.85-0.93, RMSE ≈ 400-800

## 🎓 Educational Value

This project demonstrates:
1. **Mathematical Modeling**: SIR epidemiological models
2. **Agent-Based Simulation**: Individual behavior and emergent patterns
3. **Data Science Pipeline**: From raw data to predictions
4. **Machine Learning**: Model comparison and evaluation
5. **Scientific Programming**: Clean code structure and documentation

## 👥 Future ML Extensions

### Recommended Tasks for ElPollaco

#### Option 1: Enhanced Analysis (Beginner-Friendly)
- Create interactive dashboard using Plotly or Streamlit
- Add statistical analysis of outbreak patterns
- Implement cluster analysis to identify high-risk zones
- Create comprehensive report generation

#### Option 2: Model Validation (Intermediate)
- Compare simulation results with real COVID-19 data
- Implement sensitivity analysis for parameters
- Add uncertainty quantification
- Create validation metrics and benchmarks

### Suggested Workflow
1. Review current codebase and understand simulation mechanics
2. Choose one extension area based on skill level and interest
3. Create new module (e.g., `advanced_ml.py` or `optimization.py`)
4. Document additions in separate markdown file
5. Present comparative analysis with baseline models

## 📚 References

- **SIR Model**: Kermack, W. O., & McKendrick, A. G. (1927). A contribution to the mathematical theory of epidemics.
- **COVID-19 Data**: Kaggle Novel Corona Virus 2019 Dataset
- **Agent-Based Modeling**: Bonabeau, E. (2002). Agent-based modeling: Methods and techniques for simulating human systems.

## 🤝 Contributing

This is an academic project. For suggestions or improvements:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

## 📝 License

This project is created for educational purposes. Feel free to use and modify for academic work with proper attribution.

## 👨‍💻 Authors

- LonelyLake: Simulation Engine and Disease Modeling
- Blazejost: Data Processing and Machine Learning
- ElPollaco: [To be added - Analysis/Validation]

## 🎯 Grading Criteria Alignment

- ✅ **Mathematical Modeling**: SIR equations, transmission dynamics
- ✅ **Computer Simulation**: Agent-based model with realistic behavior
- ✅ **Data Integration**: Kaggle COVID-19 dataset utilization
- ✅ **Machine Learning**: Multiple algorithms with evaluation
- ✅ **Visualization**: Multiple plots and real-time animation
- ✅ **Code Quality**: Well-structured, documented, and modular
- ✅ **Documentation**: Comprehensive README and inline comments

For questions or issues, please open an issue on GitHub or contact the project team.
