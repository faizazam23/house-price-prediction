import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("=" * 50)
print("STEP 1: Data Load Karo")
print("=" * 50)

# Data load karo
with open('data/X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
    
with open('data/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
    
with open('data/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
    
with open('data/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

print(f"✅ Data loaded!")
print(f"Training: {X_train.shape} | Testing: {X_test.shape}")

print("\n" + "=" * 50)
print("STEP 2: Linear Regression Model Train Karo")
print("=" * 50)

# Model 1: Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

print("✅ Linear Regression trained!")

# Predictions
y_pred_train_lr = model_lr.predict(X_train)
y_pred_test_lr = model_lr.predict(X_test)

# Evaluation
train_r2_lr = r2_score(y_train, y_pred_train_lr)
test_r2_lr = r2_score(y_test, y_pred_test_lr)
train_rmse_lr = np.sqrt(mean_squared_error(y_train, y_pred_train_lr))
test_rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_test_lr))

print("\n📊 Linear Regression Results:")
print(f"   Training R² Score: {train_r2_lr:.4f}")
print(f"   Testing R² Score:  {test_r2_lr:.4f}")
print(f"   Training RMSE: {train_rmse_lr:.4f}")
print(f"   Testing RMSE:  {test_rmse_lr:.4f}")

print("\n" + "=" * 50)
print("STEP 3: Random Forest Model Train Karo")
print("=" * 50)

# Model 2: Random Forest
print("Training... Please wait (1-2 minutes)")
model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)

print("✅ Random Forest trained!")

# Predictions
y_pred_train_rf = model_rf.predict(X_train)
y_pred_test_rf = model_rf.predict(X_test)

# Evaluation
train_r2_rf = r2_score(y_train, y_pred_train_rf)
test_r2_rf = r2_score(y_test, y_pred_test_rf)
train_rmse_rf = np.sqrt(mean_squared_error(y_train, y_pred_train_rf))
test_rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_test_rf))

print("\n📊 Random Forest Results:")
print(f"   Training R² Score: {train_r2_rf:.4f}")
print(f"   Testing R² Score:  {test_r2_rf:.4f}")
print(f"   Training RMSE: {train_rmse_rf:.4f}")
print(f"   Testing RMSE:  {test_rmse_rf:.4f}")

print("\n" + "=" * 50)
print("STEP 4: Best Model Select Karo")
print("=" * 50)

print("\n🏆 Model Comparison:")
print(f"Linear Regression R² Score: {test_r2_lr:.4f}")
print(f"Random Forest R² Score:     {test_r2_rf:.4f}")

if test_r2_rf > test_r2_lr:
    print("\n✅ Best Model: Random Forest")
    best_model = model_rf
    best_name = "Random Forest"
else:
    print("\n✅ Best Model: Linear Regression")
    best_model = model_lr
    best_name = "Linear Regression"

print("\n" + "=" * 50)
print("STEP 5: Best Model Save Karo")
print("=" * 50)

# Save best model
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"✅ {best_name} model saved!")
print("   File: models/best_model.pkl")

print("\n✅ Model Training Complete!")