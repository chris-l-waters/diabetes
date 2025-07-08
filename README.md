# Diabetes Hospital Readmission Prediction

A comprehensive machine learning analysis comparing six different models for predicting hospital readmission in diabetic patients using the UCI diabetes dataset. This project implements the complete data science pipeline from data cleaning through model comparison and interpretability analysis.

## Dataset
This project is built on the Diabetes 130-US hospitals dataset from the UCI Machine Learning Repository*, one of the most well-known real-world clinical datasets available for public machine learning research.

Collected over a 10-year period (1999–2008) from 130 U.S. hospitals, the dataset contains detailed information on over 100,000 hospital encounters for diabetic patients. Each row represents a single hospital visit and includes:
- Demographics: age, gender, race
- Clinical indicators: primary and secondary diagnoses, comorbidities, number of procedures
- Treatment details: medication changes, diabetes management actions
- Administrative data: admission source, discharge disposition, length of stay
- Outcome variable: whether the patient was readmitted within 30 days, after 30 days, or not at all

This dataset has become a common benchmark for exploring clinical risk prediction, readmission modeling, and healthcare resource optimization — making it an ideal foundation for comparing a variety of machine learning techniques in a practical, high-stakes setting.

***
*<sub>https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008</sub>
## Project Structure

```
diabetes/
├── *data/                               # Raw and processed datasets (need to provide)
├── *models/                             # Trained model artifacts and SHAP data (will be written)
├── notebooks/                           # Analysis notebooks
│   ├── p00_portfolio_cleaning.ipynb     # Data cleaning and validation
│   ├── p02_modeling_preprocessing.ipynb # Feature engineering and preprocessing
│   ├── p03_logreg.ipynb                 # Logistic regression models
│   ├── p04_xgboost.ipynb                # XGBoost implementation
│   ├── p05_lgbm.ipynb                   # LightGBM implementation
│   ├── p06_random_Forest.ipynb          # Random Forest implementation
│   ├── p07_neural_net.ipynb             # Neural network with KerasTuner
│   ├── p10_graphical_exploration.ipynb  # Exploratory data analysis
│   └──  p20_model_comparison.ipynb      # Comprehensive model comparison
├── requirements.txt                     # Python dependencies
└── requirements-macos.txt               # Dependencies for apple silicon macs
```

## Models Implemented

Six machine learning models with hyperparameter optimization:

1. **Logistic Regression (Lasso)** - Optimized with Optuna
2. **Logistic Regression (ElasticNet)** - Optimized with Optuna
2. **XGBoost** - 400 trials with Optuna optimization
3. **LightGBM** - 200 trials with Optuna optimization  
4. **Random Forest** - 200 trials with Optuna optimization
5. **Neural Network** - Optimized with KerasTuner

## Key Features

### Data Processing
- Comprehensive data cleaning with validation of ICD-9 diagnosis codes
- Feature engineering including medical specialty hierarchies
- Preprocessing pipelines with categorical encoding and numerical scaling
- Proper train/test split maintaining data integrity

### Model Training
- Hyperparameter optimization using Optuna for tree-based and linear models
- KerasTuner for neural network architecture search
- 5-fold cross-validation with consistent random states

### Analysis and Comparison
- Performance comparison across multiple metrics (accuracy, precision, recall, F1, ROC AUC, PR AUC)
- Threshold optimization using Youden's J statistic
- SHAP interpretability analysis for all models
- Error pattern analysis and model agreement assessment
- Feature importance comparison across different model types
- Deployment readiness evaluation (training time, model size)

### Visualizations
- ROC and Precision-Recall curves with optimal thresholds
- Confusion matrices with shared color scales
- SHAP summary plots and waterfall charts for individual predictions
- Feature distribution analysis for misclassified cases
- Model prediction agreement heatmaps

## Results Summary

**Best Performing Models:**
1. LightGBM (1.83 average rank) - Best overall performance
2. Random Forest (2.83 average rank) - Good balance of performance and interpretability
3. XGBoost (3.33 average rank) - Strong performance with reasonable complexity

**Deployment Considerations:**
- **Best for Production**: LightGBM (fast training, small size, excellent performance)
- **Most Interpretable**: Logistic models (smallest size, fastest inference)
- **Least Practical**: Neural Network (1.2GB model size, complex deployment)

## Installation and Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run notebooks in sequence:**
   - Start with `p00_portfolio_cleaning.ipynb` for data preparation
   - Continue through modeling notebooks (`p03-p07`)
   - View comprehensive comparison in `p20_model_comparison.ipynb`

3. **Key outputs:**
   - Cleaned dataset: `data/diabetes_clean.pkl`
   - Trained models: `models/` directory
   - Model metrics: `models/fits_pickle_*.pkl` files

## Technical Implementation

- **Preprocessing**: Scikit-learn pipelines with categorical encoding and scaling
- **Model Training**: Optuna for hyperparameter optimization with early stopping
- **Evaluation**: Comprehensive metrics including calibration and threshold optimization
- **Interpretability**: SHAP analysis for all model types including ensemble methods
- **Reproducibility**: Fixed random seeds and version-controlled model artifacts

## Model Performance

| Model | ROC AUC | Training Time | Model Size | Deployment Ready |
|-------|---------|---------------|------------|------------------|
| LightGBM | 0.714 | 10m | 1.9 MB | Excellent |
| Random Forest | 0.697 | 34m | 6.3 MB | Moderate |
| XGBoost | 0.705 | 7m | 3.8 MB | Good |
| Neural Network | 0.660 | 34m | 1.2 GB | Poor |
| Logistic Lasso | 0.653 | 17m | 47 KB | Excellent |
| Elastic Net | 0.653 | >60m | 47 KB | Slow Training |

This analysis demonstrates that ensemble methods (LightGBM, Random Forest, XGBoost) significantly outperform linear and neural network approaches for this healthcare prediction task, while maintaining practical deployment characteristics.