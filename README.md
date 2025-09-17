# Binary Classification with Bank Marketing Dataset

![img.png](imgs/header.png)


## 🎯 Project Overview

This repository contains a **complete, professional machine learning solution** for binary classification using a bank marketing dataset. The entire pipeline is contained in a single, well-documented notebook that combines advanced feature engineering, automated hyperparameter optimization, and ensemble learning. 

The goal is to predict whether a client will subscribe to a term deposit based on various demographic, financial, and campaign-related features.

### 🧹 **Clean, Professional Structure**
This repository has been streamlined to contain only the essential, production-ready code. All exploratory work, iterations, and experimental approaches have been consolidated into one comprehensive, professional notebook. This ensures:
- **Single source of truth** for the complete ML pipeline
- **Easy maintenance** and understanding
- **Professional presentation** suitable for portfolio showcase
- **Self-contained solution** with no external dependencies

## 🏆 Key Results

- **Final Model**: Ensemble of LightGBM, XGBoost, and CatBoost
- **Performance**: Optimized through automated hyperparameter tuning with Optuna (50+ trials per model)
- **Architecture**: Complete ML pipeline with advanced feature engineering (40+ engineered features)
- **Validation**: Robust cross-validation with stratified sampling
- **Code Quality**: Professional, documented, and maintainable codebase
- **Self-Contained**: Single notebook contains entire pipeline from EDA to final predictions

## 📚 Development Process

This project evolved through multiple iterations and approaches:

1. **Initial EDA and Basic Modeling** - Comprehensive data exploration and baseline models
2. **Advanced Feature Engineering** - Domain-specific transformations and encoding strategies  
3. **XGBoost Implementation** - Robust gradient boosting with careful preprocessing
4. **LightGBM with Advanced Features** - Native categorical handling with sophisticated features
5. **Multi-Model Optimization** - Automated hyperparameter tuning and ensemble learning

All these approaches have been **consolidated and refined** into the final professional notebook, incorporating the best techniques from each iteration while maintaining clean, production-ready code.

## 🔧 Technical Approach

### Advanced Feature Engineering
- **Domain-specific transformations** for banking/marketing context
- **Target encoding** with cross-validation to prevent leakage
- **Cyclical encoding** for temporal features (month, seasonality)
- **Interaction features** between related variables
- **Financial indicators** and contact history analysis

### Multi-Model Ensemble
- **LightGBM**: Fast gradient boosting with native categorical support
- **XGBoost**: Robust performance with advanced regularization
- **CatBoost**: Superior categorical handling with minimal preprocessing
- **Automated hyperparameter optimization** using Optuna (50+ trials per model)
- **Optimized ensemble weights** for maximum performance

### Robust Validation
- **Stratified K-Fold cross-validation** (5 folds)
- **Out-of-fold predictions** for proper ensemble training
- **No data leakage** through fold-safe encodings
- **Early stopping** to prevent overfitting

## 📁 Repository Structure

```
├── binary_classification.ipynb  # Complete ML pipeline (main notebook)
├── data/
│   ├── train.csv                                   # Training dataset
│   ├── test.csv                                    # Test dataset
│   ├── bank-full.csv                              # Additional training data
│   └── sample_submission.csv                      # Submission format
├── imgs/                                           # Project images
├── README.md                                       # Original README
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install lightgbm xgboost catboost optuna tqdm
```

### Run the Complete Analysis
1. Open `binary_classification.ipynb`
2. Execute all cells sequentially (estimated runtime: 30-60 minutes)
3. The notebook will:
   - Perform comprehensive EDA
   - Engineer advanced features
   - Optimize hyperparameters for 3 models using Optuna
   - Train ensemble and generate predictions
4. Final predictions will be saved to `final_ensemble_predictions_binary.csv`

## 📊 Methodology Highlights

### 1. Comprehensive EDA
- Data quality assessment and missing value analysis
- Feature distribution analysis with outlier detection
- Target variable correlation and class imbalance handling
- Professional visualizations with business insights

### 2. Advanced Feature Engineering
```python
# Example: Domain-specific feature creation
def add_domain_features(df):
    # Contact history features
    df['contacted_before'] = (df['pdays'] != 999).astype(int)
    df['contact_intensity'] = df['campaign'] / (df['pdays'] + 1)
    
    # Temporal cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    
    # Financial indicators
    df['has_positive_balance'] = (df['balance'] > 0).astype(int)
    
    return df
```

### 3. Automated Model Optimization
```python
# Optuna hyperparameter optimization
def tune_lightgbm(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        # ... additional parameters
    }
    return cross_validate_score(params)

study = optuna.create_study(direction="maximize")
study.optimize(tune_lightgbm, n_trials=50)
```

### 4. Ensemble Learning
- Out-of-fold prediction generation for each model
- Automated blend weight optimization using Optuna
- Final ensemble combining strengths of all models

## 💡 Business Impact

The resulting model enables:
- **Targeted Marketing**: Focus resources on high-probability prospects
- **Cost Optimization**: Reduce wasted spend on unlikely conversions  
- **Strategic Insights**: Understand key factors driving customer decisions
- **Performance Monitoring**: Track campaign effectiveness over time

## 📈 Performance Metrics

- **Primary Metric**: ROC-AUC (Area Under the Receiver Operating Characteristic curve)
- **Validation Strategy**: 5-fold stratified cross-validation
- **Ensemble Improvement**: Consistently outperforms individual models
- **Robustness**: Stable performance across different data splits

## 🔬 Key Findings

From feature importance analysis:
1. **Contact Duration**: Most predictive single feature
2. **Previous Campaign Outcome**: Strong indicator of success
3. **Seasonal Patterns**: Q4 and summer months show different conversion rates
4. **Demographics**: Age and job type significantly influence decisions
5. **Financial Status**: Account balance and loan status are key factors

## 🎓 Educational Value

This repository demonstrates:
- **Professional ML Pipeline Development**
- **Advanced Feature Engineering Techniques**
- **Automated Hyperparameter Optimization**
- **Ensemble Learning Best Practices**
- **Proper Cross-Validation and Model Selection**
- **Business-Focused Model Interpretation**

## 📧 Contact

This project showcases professional machine learning development practices and can serve as a template for similar binary classification problems in business contexts.

---
*Built with Python, scikit-learn, LightGBM, XGBoost, CatBoost, and Optuna*
