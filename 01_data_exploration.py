import pandas as pd
import numpy as np

print("=" * 50)
print("STEP 1: Data Load Karo")
print("=" * 50)

# Data load karo
data = pd.read_csv('data/house_prices.csv')

print(f"\n✅ Data Load Ho Gya!")
print(f"Rows: {len(data)}")
print(f"Columns: {len(data.columns)}")

print("\n" + "=" * 50)
print("STEP 2: Pehli 5 Rows Dekho")
print("=" * 50)
print(data.head())

print("\n" + "=" * 50)
print("STEP 3: Columns Aur Types")
print("=" * 50)
print(data.dtypes)

print("\n" + "=" * 50)
print("STEP 4: Missing Values Check Karo")
print("=" * 50)
missing = data.isnull().sum()
print(f"Total Missing Values: {missing.sum()}")
if missing.sum() == 0:
    print("✅ Koi missing values nahi! Data clean hai!")

print("\n" + "=" * 50)
print("STEP 5: Price Statistics")
print("=" * 50)
print(f"Min Price: ${data['Price'].min():.2f}")
print(f"Max Price: ${data['Price'].max():.2f}")
print(f"Average Price: ${data['Price'].mean():.2f}")
print(f"Median Price: ${data['Price'].median():.2f}")

print("\n✅ Data Exploration Complete!")