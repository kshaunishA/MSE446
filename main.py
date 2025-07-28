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
import numpy as np

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
mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
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

# ---------------------
# 🔬 Reliability Testing Section
# ---------------------
print("\n" + "="*50)
print("🔬 RELIABILITY TESTING")
print("="*50)

# 📊 Reliability Test Results Storage
reliability_results = {model: {'mae_scores': [], 'r2_scores': []} for model in results.keys()}

# Test with different random seeds
seeds = [42, 123, 456]  # Reduced from 5 to 3
test_sizes = [0.2, 0.25]  # Reduced from 4 to 2

print(f"Testing with {len(seeds)} different seeds and {len(test_sizes)} different test sizes...")
print(f"Total combinations: {len(seeds) * len(test_sizes)}")

for seed in seeds:
    for test_size in test_sizes:
        # Split data with current seed and test size
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
            X_scaled, y, test_size=test_size, random_state=seed
        )
        
        # Test each model
        for model_name in results.keys():
            if model_name == 'Linear':
                model = LinearRegression()
            elif model_name == 'XGBoost':
                model = XGBRegressor(random_state=seed)
            elif model_name == 'MLP':
                model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, random_state=seed)  # Reduced complexity
            elif model_name == 'KNN':
                model = KNeighborsRegressor(n_neighbors=5)
            
            # Train and predict
            model.fit(X_train_temp, y_train_temp)
            y_pred_temp = model.predict(X_test_temp)
            
            # Store metrics
            mae = mean_absolute_error(y_test_temp, y_pred_temp)
            r2 = r2_score(y_test_temp, y_pred_temp)
            
            reliability_results[model_name]['mae_scores'].append(mae)
            reliability_results[model_name]['r2_scores'].append(r2)

# 📊 Calculate reliability statistics
print("\n📈 RELIABILITY STATISTICS:")
print("-" * 40)

for model_name in reliability_results.keys():
    mae_scores = reliability_results[model_name]['mae_scores']
    r2_scores = reliability_results[model_name]['r2_scores']
    
    mae_mean = np.mean(mae_scores)
    mae_std = np.std(mae_scores)
    mae_cv = mae_std / mae_mean  # Coefficient of variation
    
    r2_mean = np.mean(r2_scores)
    r2_std = np.std(r2_scores)
    
    print(f"\n🔸 {model_name}:")
    print(f"   MAE: {mae_mean:.4f} ± {mae_std:.4f} (CV: {mae_cv:.3f})")
    print(f"   R²:  {r2_mean:.4f} ± {r2_std:.4f}")
    print(f"   MAE Range: [{min(mae_scores):.4f}, {max(mae_scores):.4f}]")
    print(f"   R² Range:  [{min(r2_scores):.4f}, {max(r2_scores):.4f}]")

# 📊 Reliability Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# MAE Box Plot
mae_data = [reliability_results[model]['mae_scores'] for model in reliability_results.keys()]
ax1.boxplot(mae_data, tick_labels=list(reliability_results.keys()))
ax1.set_title('MAE Reliability Across Different Seeds & Test Sizes')
ax1.set_ylabel('Mean Absolute Error')
ax1.grid(True, alpha=0.3)

# R² Box Plot
r2_data = [reliability_results[model]['r2_scores'] for model in reliability_results.keys()]
ax2.boxplot(r2_data, tick_labels=list(reliability_results.keys()))
ax2.set_title('R² Reliability Across Different Seeds & Test Sizes')
ax2.set_ylabel('R² Score')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 🎯 Reliability Assessment
print("\n🎯 RELIABILITY ASSESSMENT:")
print("-" * 30)

# Find most reliable model (lowest coefficient of variation for MAE)
reliability_scores = {}
for model_name in reliability_results.keys():
    mae_scores = reliability_results[model_name]['mae_scores']
    mae_mean = np.mean(mae_scores)
    mae_std = np.std(mae_scores)
    cv = mae_std / mae_mean
    reliability_scores[model_name] = cv

most_reliable = min(reliability_scores, key=reliability_scores.get)
print(f"✅ Most Reliable Model: {most_reliable} (CV: {reliability_scores[most_reliable]:.3f})")

# Check if results are consistent (low CV indicates consistency)
print("\n📊 Consistency Check:")
for model, cv in reliability_scores.items():
    status = "✅ Consistent" if cv < 0.1 else "⚠️ Variable" if cv < 0.2 else "❌ Inconsistent"
    print(f"   {model}: {status} (CV: {cv:.3f})")
