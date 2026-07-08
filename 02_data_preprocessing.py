import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

print("=" * 50)
print("STEP 1: Data Load Karo")
print("=" * 50)

# Data load karo
data = pd.read_csv('data/house_prices.csv')
print(f"✅ Data loaded: {data.shape}")

print("\n" + "=" * 50)
print("STEP 2: Target (Price) aur Features Separate Karo")
print("=" * 50)

# Target variable (Price - jo predict karna hai)
y = data['Price']
print(f"Target (y) shape: {y.shape}")
print(f"Price range: {y.min():.2f} to {y.max():.2f}")

# Features (jo use karenge)
X = data.drop('Price', axis=1)
print(f"\nFeatures (X) shape: {X.shape}")
print(f"Columns: {list(X.columns)}")

print("\n" + "=" * 50)
print("STEP 3: Train-Test Split Karo (80-20)")
print("=" * 50)

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"\nTotal: {len(X_train) + len(X_test)}")

print("\n" + "=" * 50)
print("STEP 4: Data Save Karo (Pickle Format)")
print("=" * 50)

# Pickle format mein save karo
with open('data/X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)

with open('data/X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)

with open('data/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('data/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)

print("✅ Files saved:")
print("   - data/X_train.pkl")
print("   - data/X_test.pkl")
print("   - data/y_train.pkl")
print("   - data/y_test.pkl")

print("\n✅ Data Preprocessing Complete!")