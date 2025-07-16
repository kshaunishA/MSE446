# MSE446
MSE446 Project -> Testing Models on Housing Datasets

# Housing Forecasting & Decarbonization Benchmarking

## 🌍 Overview
This project benchmarks multiple machine learning models to forecast housing affordability and building decarbonization across regions. We compare **linear regressions**, **tree ensembles**, **neural nets**, and **spatial models** using public datasets from Canada and the U.S.

Why this matters:
- The global housing crisis is intensifying, with rising prices and falling availability.
- Buildings contribute ~26% of energy-related CO₂ emissions.
- Accurate, fair, and transferable predictive models are crucial for policy, finance, and climate action.

---

## 👥 Stakeholders
- **City Planners**: Improved affordability and zoning forecasts
- **PropTechs & Banks**: Better valuations and risk predictions
- **Climate Agencies**: Clearer decarbonization baselines
- **Researchers**: Transparent, reproducible benchmarks

---

## 📊 Datasets Used
- [Canadian House Prices (Kaggle)](https://www.kaggle.com/datasets/jeremylarcher/canadian-house-prices-for-top-cities)
- [Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)
- [Toronto Open Data Portal](https://open.toronto.ca/)
- [HUD User Datasets (U.S. Housing)](https://www.huduser.gov/portal/pdrdatas_landing.html)
- [Data.gov Housing Tag](https://catalog.data.gov/dataset/?tags=housing)

*See `data/README.md` for full details and licenses.*

---

## 🔧 How It Works

### 1. **Preprocessing**
- Missing value imputation
- Encoding categorical variables
- Feature scaling for numerical columns
- Spatial data joins (if applicable)

### 2. **Modeling**
- Baseline: mean predictor
- Models:
  - Linear Regression
  - Random Forest / XGBoost
  - Deep Neural Network (PyTorch / TensorFlow)
  - Spatial model (e.g. GWR, Kriging)

### 3. **Evaluation**
- Metrics:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score
- Fairness & generalizability checks across geographies
- Reproducibility ensured via fixed seeds & tracked configs

---



