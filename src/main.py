import random
import requests
import time
import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

API_KEY = 'c03098a926c34a25b6b162347242710'
LATITUDE = '13.7229936'  
LONGITUDE = '100.7728991'  
CSV_FILE = 'sensor_data_log.csv'
model = None

def get_sensor_data():  # Random data for testing
    temperature = random.uniform(20, 30)
    humidity = random.uniform(40, 100)
    pressure = random.uniform(980, 1020)
    return temperature, humidity, pressure

def fetch_historical_data():
    end_time = int(time.time())
    start_time = end_time - (3 * 60 * 60)

    url = f"http://api.weatherapi.com/v1/history.json?key={API_KEY}&q={LATITUDE},{LONGITUDE}&dt={time.strftime('%Y-%m-%d', time.localtime(start_time))}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        historical_temp = data['forecast']['forecastday'][0]['hour'][-1]['temp_c']  
        historical_humidity = data['forecast']['forecastday'][0]['hour'][-1]['humidity']  
        historical_pressure = data['forecast']['forecastday'][0]['hour'][-1]['pressure_mb']  
        return historical_temp, historical_humidity, historical_pressure
    else:
        print(f"Failed to fetch historical data. Status code: {response.status_code}, Response: {response.text}")
        return None, None, None

def log_sensor_data(temperature, humidity, pressure):
    data = pd.DataFrame({
        'timestamp': [datetime.datetime.now()],
        'temperature': [temperature],
        'humidity': [humidity],
        'pressure': [pressure]
    })
    data.to_csv(CSV_FILE, mode='a', header=not pd.io.common.file_exists(CSV_FILE), index=False)

def get_last_logged_temp():
    try:
        # Load the last row of the CSV file
        data = pd.read_csv(CSV_FILE)
        last_temp = data['temperature'].iloc[-1]  # Get the last recorded temperature
        return last_temp
    except (FileNotFoundError, IndexError):
        print("No previous log data found.")
        return None

def send_to_thingspeak(temperature, humidity, pressure, temp_change, prediction):
    print(f"Sending data - Temp: {temperature}, Humidity: {humidity}, Pressure: {pressure}")
    print(f"Temp Change: {temp_change}, Prediction: {prediction}")
    
    api_key = 'YTQF6O7N4SV0Z3AB' 
    url = f'https://api.thingspeak.com/update?api_key={api_key}'
    response = requests.post(url, params={
        'field1': temperature,
        'field2': humidity,
        'field3': pressure,
        'field4': temp_change,
        'field5': prediction
    })

    if response.status_code == 200:
        print("Data sent successfully to ThingSpeak.")
    else:
        print(f"Error sending data. Status code: {response.status_code}")

# New function to train the model using historical data
def train_model():
    global model
    try:
        data = pd.read_csv(CSV_FILE)
        X = data[['temperature', 'humidity', 'pressure']]
        y = (data['humidity'] > 70).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        print("Model trained successfully.")
    except Exception as e:
        print(f"Error training model: {e}")

def predict_rain_with_model(temperature, humidity, pressure):
    global model
    if model is None:
        print("Model is not trained yet.")
        return "Model not available for prediction"

    input_data = np.array([[temperature, humidity, pressure]])
    input_data = StandardScaler().fit_transform(input_data)
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        return "High chance of rain"
    else:
        return "Low chance of rain"

if __name__ == "__main__":
    train_model()  # Train model once when starting

    while True:
        temperature, humidity, pressure = get_sensor_data()
        historical_data = fetch_historical_data()
        
        if historical_data[0] is not None:
            last_temp = get_last_logged_temp()  # Fetch last logged temperature
            
            # Calculate the temperature change if we have a previous log
            temp_change = temperature - last_temp if last_temp is not None else 0
            
            prediction = predict_rain_with_model(temperature, humidity, pressure)
            log_sensor_data(temperature, humidity, pressure)  # Log new data
            send_to_thingspeak(temperature, humidity, pressure, temp_change, prediction)
        
        else:
            print("Skipping prediction due to failed historical data fetch.")
        
        time.sleep(15)
