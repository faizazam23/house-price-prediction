import pandas as pd
from sklearn.datasets import fetch_california_housing

print("=" * 50)
print("Dataset Download Ho Raha Hai...")
print("=" * 50)

# California Housing dataset load karo (similar to Boston)
housing = fetch_california_housing()

# DataFrame banao
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['Price'] = housing.target

print(f"\n✅ Dataset Load Ho Gya!")
print(f"Total Rows: {len(data)}")
print(f"Total Columns: {len(data.columns)}")

print("\nPehli 5 Rows:")
print(data.head())

# CSV file mein save karo
data.to_csv('data/house_prices.csv', index=False)

print("\n✅ File Save Ho Gya: data/house_prices.csv")
print("=" * 50)