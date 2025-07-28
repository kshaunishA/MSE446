# 🏠 Toronto Housing Price Index Analysis

A comprehensive machine learning analysis of Toronto's housing market using the Toronto Home Price Index dataset from Kaggle. This project implements multiple regression models with reliability testing to predict housing price indices.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Reliability Testing](#reliability-testing)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project analyzes Toronto's housing market data to predict housing price indices using various machine learning algorithms. The analysis includes comprehensive data preprocessing, model training, performance evaluation, and reliability testing to ensure robust results.

### Key Objectives:
- Download and preprocess Toronto housing market data
- Implement multiple regression models
- Evaluate model performance using various metrics
- Conduct reliability testing with different seeds and data splits
- Visualize results and model comparisons

## ✨ Features

- **Automated Data Download**: Uses KaggleHub to automatically download the latest dataset
- **Comprehensive Preprocessing**: Handles missing values, column cleaning, and feature scaling
- **Multiple ML Models**: Implements Linear Regression, XGBoost, Neural Networks, and KNN
- **Reliability Testing**: Tests models across different random seeds and data splits
- **Performance Visualization**: Generates charts and statistical summaries
- **Robust Evaluation**: Uses MAE, MSE, and R² metrics for comprehensive assessment

## 📊 Dataset

**Source**: [Toronto Home Price Index](https://www.kaggle.com/datasets/alankmwong/toronto-home-price-index) on Kaggle

**Dataset Details**:
- **Size**: 5,091 records × 17 columns
- **Time Period**: 2015 onwards
- **Features**: Various housing indices and benchmarks
- **Target**: Composite Index (CompIndex)

**Key Columns**:
- `Location`: Geographic area in Toronto
- `CompIndex`: Composite Price Index (target variable)
- `CompBenchmark`: Composite Benchmark Price
- `SFDetachBenchmark`: Single Family Detached Benchmark
- `SFAttachBenchmark`: Single Family Attached Benchmark
- `Date`: Time period of the data

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. **Clone or download the project**:
   ```bash
   # If using git
   git clone <repository-url>
   cd MSE-Project
   ```

2. **Install required dependencies**:
   ```bash
   pip install kagglehub pandas scikit-learn xgboost matplotlib numpy
   ```

3. **Verify installation**:
   ```bash
   python -c "import kagglehub, pandas, sklearn, xgboost; print('All packages installed successfully!')"
   ```

## 📖 Usage

### Quick Start

Run the complete analysis:

```bash
python main.py
```

### What the Script Does

1. **Data Download**: Automatically downloads the Toronto housing dataset
2. **Data Preprocessing**: 
   - Cleans column names
   - Handles missing values
   - Scales features
3. **Model Training**: Trains 4 different regression models
4. **Performance Evaluation**: Calculates MAE, MSE, and R² scores
5. **Reliability Testing**: Tests models with different seeds and data splits
6. **Visualization**: Generates performance charts

### Expected Output

The script will output:
- Dataset information and preprocessing steps
- Model performance metrics
- Reliability testing statistics
- Visualization charts

## 📁 Project Structure

```
MSE Project/
├── main.py              # Main analysis script
├── README.md           # This documentation
└── .cache/             # KaggleHub cache (auto-generated)
    └── kagglehub/
        └── datasets/
            └── alankmwong/
                └── toronto-home-price-index/
```

## 🤖 Models Implemented

### 1. Linear Regression
- **Type**: Simple linear model
- **Use Case**: Baseline performance
- **Advantages**: Interpretable, fast training

### 2. XGBoost
- **Type**: Gradient boosting
- **Use Case**: High-performance prediction
- **Advantages**: Handles non-linear relationships, robust

### 3. MLP Regressor (Neural Network)
- **Type**: Multi-layer perceptron
- **Use Case**: Complex pattern recognition
- **Architecture**: 32 → 16 neurons
- **Advantages**: Can capture complex relationships

### 4. K-Nearest Neighbors (KNN)
- **Type**: Instance-based learning
- **Use Case**: Local pattern recognition
- **Parameters**: k=5 neighbors
- **Advantages**: Simple, no training required

## 📈 Results

### Model Performance Comparison

| Model | MAE | MSE | R² Score |
|-------|-----|-----|----------|
| Linear | 26.19 | 1180.80 | 0.2397 |
| XGBoost | 14.09 | 475.94 | 0.6936 |
| MLP | 22.47 | 872.78 | 0.4380 |
| KNN | 11.56 | 406.55 | 0.7382 |

### Key Findings

- **Best Performance**: KNN achieved the highest R² score (0.7382)
- **Lowest Error**: KNN also had the lowest MAE (11.56)
- **Most Reliable**: MLP showed the most consistent results across different conditions

## 🔬 Reliability Testing

The project includes comprehensive reliability testing to ensure robust results:

### Testing Methodology
- **Multiple Seeds**: Tests with 3 different random seeds (42, 123, 456)
- **Varied Data Splits**: Uses different test sizes (20%, 25%)
- **Total Combinations**: 6 different test scenarios

### Reliability Metrics
- **Coefficient of Variation (CV)**: Measures consistency
- **Range Analysis**: Shows performance variability
- **Consistency Classification**: 
  - ✅ Consistent (CV < 0.1)
  - ⚠️ Variable (CV < 0.2)
  - ❌ Inconsistent (CV ≥ 0.2)

### Reliability Results

| Model | CV (MAE) | Consistency Status |
|-------|----------|-------------------|
| MLP | 0.025 | ✅ Consistent |
| Linear | 0.039 | ✅ Consistent |
| KNN | 0.050 | ✅ Consistent |
| XGBoost | 0.058 | ✅ Consistent |

## 🔧 Technical Details

### Data Preprocessing Steps

1. **Column Cleaning**: Converts column names to lowercase with underscores
2. **Missing Value Handling**: Drops rows with missing values
3. **Feature Scaling**: Uses StandardScaler for normalization
4. **Train/Test Split**: 80/20 split with random state 42

### Feature Engineering

**Selected Features**:
- `compbenchmark`: Composite benchmark price
- `sfdetachbenchmark`: Single family detached benchmark
- `sfattachbenchmark`: Single family attached benchmark

**Target Variable**:
- `compindex`: Composite price index

### Model Parameters

```python
# XGBoost
XGBRegressor(random_state=42)

# MLP Regressor
MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)

# KNN
KNeighborsRegressor(n_neighbors=5)

# Linear Regression
LinearRegression()
```

## 📊 Visualization

The project generates several visualizations:

1. **MAE Bar Chart**: Compares model performance
2. **Reliability Box Plots**: Shows performance distribution across different conditions
3. **Statistical Summaries**: Detailed performance metrics

## 🛠️ Customization

### Modifying Features

To use different features, edit the features list in the code:

```python
features = ['compbenchmark', 'sfdetachbenchmark', 'sfattachbenchmark']
```

### Adding New Models

To add a new model, follow this pattern:

```python
from sklearn.ensemble import RandomForestRegressor

# Add to models section
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results['RandomForest'] = {
    "MAE": mean_absolute_error(y_test, y_pred_rf),
    "MSE": mean_squared_error(y_test, y_pred_rf),
    "R2": r2_score(y_test, y_pred_rf)
}
```

### Adjusting Reliability Testing

Modify the testing parameters:

```python
seeds = [42, 123, 456, 789]  # Add more seeds
test_sizes = [0.15, 0.2, 0.25, 0.3]  # Add more test sizes
```

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'kagglehub'"**
   ```bash
   pip install kagglehub
   ```

2. **"No module named 'xgboost'"**
   ```bash
   pip install xgboost
   ```

3. **Convergence warnings for MLP**
   - These are normal and don't affect results
   - Can be reduced by increasing `max_iter` or simplifying architecture

4. **Memory issues with large datasets**
   - Reduce the number of reliability test combinations
   - Use smaller neural network architectures

### Performance Optimization

- **Faster execution**: Reduce the number of seeds and test sizes
- **Better accuracy**: Try different feature combinations
- **More reliable results**: Increase the number of reliability tests

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Alankmwong](https://www.kaggle.com/alankmwong) for providing the Toronto Home Price Index dataset
- **Kaggle**: For hosting the dataset
- **Scikit-learn**: For the machine learning framework
- **XGBoost**: For the gradient boosting implementation

## 📞 Contact

For questions or support, please open an issue in the repository or contact the project maintainer.

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Python Version**: 3.8+
