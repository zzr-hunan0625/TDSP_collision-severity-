import requests
import pandas as pd

def fetch_weather_data(start_date = "2019-12-01", end_date = "2024-12-01", threshold= 2.5):
    """
    Fetch daily precipitation data for multiple locations and classify it based on a threshold.

    Args:
        start_date (str): Start date in the format 'YYYY-MM-DD'.
        end_date (str): End date in the format 'YYYY-MM-DD'.
        locations (list): List of dictionaries containing 'latitude', 'longitude', and 'borough' keys.
        threshold (float): Precipitation threshold to classify the data (default is 2.5).

    Returns:
        pd.DataFrame: A DataFrame containing 'date', 'precipitation_sum' (binary), and 'borough'.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params_base = {
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": "America/New_York"
    }
    weather_data = []
    locations = [
        {"latitude": 40.78, "longitude": -73.97, "borough": 1},
        {"latitude": 40.67, "longitude": -73.94, "borough": 3},
        {"latitude": 40.73, "longitude": -73.79, "borough": 4},
        {"latitude": 40.84, "longitude": -73.86, "borough": 2},
        {"latitude": 40.58, "longitude": -74.15, "borough": 5}
    ]
    for loc in locations:
        params = params_base.copy()
        params.update({
            "latitude": loc["latitude"],
            "longitude": loc["longitude"]
        })

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            daily = data["daily"]
            
            daily_dates = pd.to_datetime(daily["time"])
            daily_precipitation = daily["precipitation_sum"]
            daily_precipitation = [x if x is not None else 0 for x in daily_precipitation]

            for date, precipitation in zip(daily_dates, daily_precipitation):
                weather_data.append({
                    "date": date,
                    "precipitation_sum": 1 if precipitation > threshold else 0,
                    "borough": loc["borough"]
                })
        else:
            print(f"Failed to fetch data for borough {loc['borough']} (Status: {response.status_code})")

    return pd.DataFrame(weather_data)


