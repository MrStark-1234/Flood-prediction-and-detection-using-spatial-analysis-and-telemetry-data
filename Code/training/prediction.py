import csv
import datetime
import pickle
import requests
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def get_data(lat, lon):
    api_key = os.getenv('WEATHER_API_KEY')
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}?unitGroup=us&key={api_key}&contentType=json"
    
    # Send the GET request
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print(response.text)  # This will print the raw response, useful for debugging
        return None

    try:
        data = response.json()['days']
    except (requests.exceptions.JSONDecodeError, KeyError) as e:
        print("Error: Could not decode JSON or 'days' key missing.")
        print(response.text)  # Print the response to help debug
        return None

    final = [0, 0, 0, 0]  # [avg_temp, max_temp, avg_wind_speed, avg_cloud_cover, total_precip, avg_humidity]

    # Accumulate the weather data
    for day in data:
        final[0] += day.get('temp', 0)
        final[1] = max(final[1], day.get('tempmax', -float('inf')))
        final[2] += day.get('windspeed', 0)
        # final[3] += day.get('cloudcover', 0)
        # final[4] += day.get('precip', 0)
        final[3] += day.get('humidity', 0)

    days_count = len(data)
    if days_count > 0:
        final[0] /= days_count
        final[2] /= days_count
        final[3] /= days_count
        # final[5] /= days_count

    return final

# Example usage
latitude = 37.7749
longitude = -122.4194
weather_data = get_data(latitude, longitude)

if weather_data:
    print(weather_data)
else:
    print("Failed to retrieve weather data.")

def testConnection():
    return "yo"
