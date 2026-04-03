import pandas as pd

# Load dataset
df = pd.read_csv("traffic.csv")

# Convert DateTime column
df['datetime'] = pd.to_datetime(df['datetime'])

# Extract features
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day
df['month'] = df['datetime'].dt.month

print(df.head())