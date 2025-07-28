# 📥 Import and Download Dataset
import kagglehub
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Download dataset
path = kagglehub.dataset_download("alankmwong/toronto-home-price-index")
print("Path to dataset files:", path)

# Step 2: Locate the CSV file
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV files found in the dataset directory.")
csv_path = os.path.join(path, csv_files[0])

# Step 3: Load CSV into DataFrame
df = pd.read_csv(csv_path)
print("Initial shape:", df.shape)
print("Initial columns:\n", df.columns)

# 🧹 Data Preprocessing
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
df = df.dropna()
print("New shape after dropping NA:", df.shape)

# Convert 'month' to numeric if present
if 'month' in df.columns:
    df['month'] = pd.to_datetime(df['month'])
    df['month_num'] = df['month'].dt.month + 12 * (df['month'].dt.year - df['month'].dt.year.min())

# Step 4: Choose features and target
features = ['compbenchmark', 'sfdetachbenchmark', 'sfattachbenchmark']
if 'month_num' in df.columns:
    features.append('month_num')

target_col = 'compindex'  # Target variable

X = df[features]
y = df[target_col]

# Step 5: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------------
# 🔮 Modeling Section
# ---------------------
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 📊 Dictionary to hold model results
results = {}

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results['Linear'] = {
    "MAE": mean_absolute_error(y_test, y_pred_lr),
    "MSE": mean_squared_error(y_test, y_pred_lr),
    "R2": r2_score(y_test, y_pred_lr)
}

# 2. XGBoost
xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
results['XGBoost'] = {
    "MAE": mean_absolute_error(y_test, y_pred_xgb),
    "MSE": mean_squared_error(y_test, y_pred_xgb),
    "R2": r2_score(y_test, y_pred_xgb)
}

# 3. MLP Regressor (Deep Net)
mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
results['MLP'] = {
    "MAE": mean_absolute_error(y_test, y_pred_mlp),
    "MSE": mean_squared_error(y_test, y_pred_mlp),
    "R2": r2_score(y_test, y_pred_mlp)
}

# 4. KNN Regressor (Spatial Proxy)
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
results['KNN'] = {
    "MAE": mean_absolute_error(y_test, y_pred_knn),
    "MSE": mean_squared_error(y_test, y_pred_knn),
    "R2": r2_score(y_test, y_pred_knn)
}

# 🖨️ Print Results
print("\n🔍 Model Performance:")
for model, metrics in results.items():
    print(f"\n🔸 {model}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# 📊 MAE Bar Chart
mae_scores = [results[m]["MAE"] for m in results]
models = list(results.keys())

plt.figure(figsize=(8, 5))
plt.bar(models, mae_scores, color='skyblue')
plt.title("Model Comparison - MAE")
plt.ylabel("Mean Absolute Error")
plt.xlabel("Model")
plt.show()
