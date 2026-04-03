import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

def train_weather_model():
    df = pd.read_csv("weather.csv")

    df['datetime'] = pd.to_datetime(df['datetime'])

    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.weekday

    # cyclic features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['area'] = df['area'].astype('category').cat.codes
    df['weather_condition'] = df['weather_condition'].astype('category').cat.codes

    # ✅ 9 FEATURES
    X = df[[
        'hour','day','month',
        'day_of_week',
        'area',
        'hour_sin','hour_cos',
        'month_sin','month_cos'
    ]]

    y = df[[
        'temperature_C',
        'humidity_%',
        'rainfall_mm',
        'wind_speed_kmph',
        'weather_condition'
    ]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor(n_estimators=20, max_depth=8, n_jobs=-1)
    model.fit(X_train, y_train)

    with open("weather_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Weather model trained (9 features)")

if __name__ == "__main__":
    train_weather_model()