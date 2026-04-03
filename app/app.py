from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import requests
from datetime import datetime
import os

app = Flask(__name__)

GOOGLE_API_KEY = "AIzaSyBWz8FvOq4Azo2QZujcxQFMzJrfoa2TmsY"

# ✅ Load models
try:
    model = pickle.load(open("model.pkl", "rb"))
    print("✅ Traffic model loaded")
except:
    model = None

try:
    weather_model = pickle.load(open("weather_model.pkl", "rb"))
    print("✅ Weather model loaded")
except:
    weather_model = None


# 🌍 Geocoding
def get_lat_lng(place):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={place}&key={GOOGLE_API_KEY}"
    res = requests.get(url).json()

    if res.get('status') != 'OK':
        return None

    loc = res['results'][0]['geometry']['location']
    return loc['lat'], loc['lng']


# 🌦️ Weather
def get_weather_features(hour, day, month, area):

    day_of_week = datetime(2024, month, day).weekday()

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    features = np.array([[hour, day, month, day_of_week, area,
                          hour_sin, hour_cos, month_sin, month_cos]])

    if weather_model:
        pred = weather_model.predict(features)[0]
    else:
        pred = [30, 60, 0, 10, 1]

    return float(pred[0]), float(pred[1]), max(0, float(pred[2])), float(pred[3]), int(round(pred[4])), \
           ["Clear","Cloudy","Rainy","Foggy","Storm"][int(round(pred[4])) % 5]


# 🚀 MULTI ROUTE ML
def get_best_predicted_route(source, dest, hour, day, month):

    url = f"https://maps.googleapis.com/maps/api/directions/json?origin={source}&destination={dest}&alternatives=true&mode=driving&key={GOOGLE_API_KEY}"
    res = requests.get(url).json()

    if res['status'] != 'OK':
        return None

    routes_data = []

    for route in res['routes']:

        leg = route['legs'][0]

        distance = leg['distance']['value'] / 1000
        base_time = leg['duration']['value'] / 60

        area = int(distance) % 5

        # 🚦 traffic logic
        if 7 <= hour <= 10 or 17 <= hour <= 20:
            vehicle_count = 300
        elif 11 <= hour <= 16:
            vehicle_count = 180
        else:
            vehicle_count = 80

        # 🚗 speed
        if vehicle_count > 250:
            avg_speed_kmph = 20
        elif vehicle_count > 150:
            avg_speed_kmph = 30
        else:
            avg_speed_kmph = 45

        is_peak_hour = 1 if (7 <= hour <= 10 or 17 <= hour <= 20) else 0
        day_of_week = datetime(2024, month, day).weekday()

        event_flag = 0
        accidents_est = 0
        distance_feature = distance

        # 🌦️ weather
        temp, humidity, rain, wind, weather_encoded, weather_text = get_weather_features(
            hour, day, month, area
        )

        # cyclic
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        features = np.array([[
            hour, day, month,
            month_sin, month_cos,
            area,
            distance_feature,
            vehicle_count,
            avg_speed_kmph,
            is_peak_hour,
            day_of_week,
            temp,
            humidity,
            rain,
            wind,
            weather_encoded,
            event_flag,
            accidents_est
        ]])

        congestion = model.predict(features)[0] if model else 50

        # 🗺️ STEP COLORS (CORRECT POSITION)
        steps_data = []

        for step in leg['steps']:
            step_distance = step['distance']['value'] / 1000
            step_congestion = congestion + (step_distance * 2)

            if step_congestion < 30:
                color = "blue"
            elif step_congestion < 70:
                color = "yellow"
            else:
                color = "red"

            steps_data.append({
                "polyline": step['polyline']['points'],
                "color": color,
                "start": step['start_location'],
                "end": step['end_location']
            })

        predicted_time = (distance / avg_speed_kmph) * 60 * (1 + congestion / 100)

        routes_data.append({
            "time": predicted_time,
            "distance": round(distance, 2),
            "base_time": base_time,
            "weather": weather_text,
            "temp": temp,
            "rain": rain,
            "steps": steps_data
        })

    # ✅ SORT + TOP 3
    routes_data = sorted(routes_data, key=lambda x: x['time'])
    return routes_data[:3]


# 🚀 MAIN PAGE
@app.route("/")
def home():
    return render_template("index.html")


# 🚀 PREDICT (MULTI ROUTE)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        hour = int(data["hour"])
        day = int(data["day"])
        month = int(data["month"])

        src = get_lat_lng(data["source"])
        dst = get_lat_lng(data["destination"])

        if not src or not dst:
            return jsonify({"error": "Location error"})

        src_str = f"{src[0]},{src[1]}"
        dst_str = f"{dst[0]},{dst[1]}"

        routes = get_best_predicted_route(src_str, dst_str, hour, day, month)

        return jsonify({
            "routes": routes
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)})


# 🚀 GRAPH
@app.route("/predict_day", methods=["POST"])
def predict_day():
    try:
        data = request.get_json()

        day = int(data["day"])
        month = int(data["month"])

        hours = list(range(24))
        congestion_values = []

        for hour in hours:
            congestion = np.random.randint(10, 90)  # simplified
            congestion_values.append(congestion)

        # find best hour
        user_hour = int(data.get("hour", 23))

        best_hour = 0
        min_congestion = float('inf')

        for h in range(user_hour + 1):
            if congestion_values[h] < min_congestion:
                min_congestion = congestion_values[h]
                best_hour = h
                
        return jsonify({
            "hours": hours,
            "congestion": congestion_values,
            "best_hour": best_hour
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

