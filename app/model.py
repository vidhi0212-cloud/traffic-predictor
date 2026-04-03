import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

def train_model():
    traffic = pd.read_csv("traffic.csv")
    weather = pd.read_csv("weather.csv")

    traffic['datetime'] = pd.to_datetime(traffic['datetime'])
    weather['datetime'] = pd.to_datetime(weather['datetime'])

    # ✅ merge
    df = pd.merge(traffic, weather, on=['datetime','area'])

    # time features
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.weekday

    # cyclic features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # encoding
    df['area'] = df['area'].astype('category').cat.codes
    df['distance_feature'] = df['vehicle_count'] / df['avg_speed_kmph']
    df['weather_condition'] = df['weather_condition'].astype('category').cat.codes

    # ✅ FEATURES (FINAL 17)
    X = df[[
        'hour','day','month',
        'month_sin','month_cos',
        'area','distance_feature',
        'vehicle_count','avg_speed_kmph',
        'is_peak_hour','day_of_week',
        'temperature_C','humidity_%',
        'rainfall_mm','wind_speed_kmph',
        'weather_condition',
        'event_flag','accidents_est'
    ]]

    y = df['congestion_%']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ FINAL model trained (18 features matched)")

if __name__ == "__main__":
    train_model()