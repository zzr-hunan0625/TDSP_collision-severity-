"""
File: pre-processing.py
Description:
This script processes collision data for New York City by integrating weather and street attribute data,
labeling incidents by severity, and preparing the data for further analysis. The main steps include:

1. Fetching collision data from the NYC open data API.
2. Cleaning and filtering the collision data.
3. Enriching the dataset with street-level attributes using the LION map.
4. Labeling incidents based on severity (e.g., no injury, injury, fatality).
5. Adding weather data from historical weather archives.
6. Creating and saving a comprehensive dataset for machine learning models.
7. Visualizing the processed data with a stacked bar chart.

Dependencies:
- pandas
- numpy
- matplotlib
- requests
- retry_requests
- custom modules: `lionmap_concat` and `fetch_weather_data`

Ensure the file `lion_name.csv` (LION map data) and any dependencies are in the working directory.

"""

import requests
from retry_requests import retry
import pandas as pd
from lionmap_concat import lionmap_concat
import numpy as np
import matplotlib.pyplot as plt
from fetch_weather_data import fetch_weather_data
import os

# Get the current working directory
current_working_directory = os.getcwd()
location = os.path.join(current_working_directory, "processed_data.csv")

# Step 1: Fetch collision data from NYC Open Data API
url = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"
params = {
    "$limit": 10000,  # Limit the number of rows fetched
    "$offset": 0     # Start fetching from the first row
}
response = requests.get(url, params=params)
if response.status_code == 200:
    data = response.json()
else:
    print(f"Request failed with status code: {response.status_code}")
df = pd.json_normalize(data)
df['crash_date'] = pd.to_datetime(df['crash_date'], errors='coerce')  # Convert crash date to datetime

# Step 2: Clean and filter collision data
df_cleaned = df[['number_of_persons_injured', 'number_of_persons_killed', 'on_street_name', 
                 'latitude', 'longitude', 'contributing_factor_vehicle_1', 
                 'vehicle_type_code1', 'vehicle_type_code2', 'crash_date']]
df_cleaned['number_of_persons_killed'].fillna(0, inplace=True)  # Replace missing fatalities with 0
df_cleaned['number_of_persons_injured'].fillna(0, inplace=True)  # Replace missing injuries with 0
df_cleaned = df_cleaned.dropna(subset=['on_street_name', 'latitude', 'longitude'])  # Drop rows with missing key data

# Step 3: Integrate with LION map attributes
df_lion = lionmap_concat(df_cleaned)
rows_dropped = len(df_cleaned) - len(df_lion)  # Calculate the number of rows dropped during merging

# Step 4: Label collision severity levels
df_lion['number_of_persons_injured'] = pd.to_numeric(df_lion['number_of_persons_injured'], errors='coerce')
df_lion['number_of_persons_killed'] = pd.to_numeric(df_lion['number_of_persons_killed'], errors='coerce')
conditions = [
    (df_lion['number_of_persons_injured'] == 0) & (df_lion['number_of_persons_killed'] == 0),
    (df_lion['number_of_persons_injured'] > 0) & (df_lion['number_of_persons_killed'] == 0),
    (df_lion['number_of_persons_killed'] > 0)
]
choices = [1, 2, 3]  # 1: No injury, 2: Injury, 3: Fatality
df_lion['incident_severity'] = np.select(conditions, choices, default=0)

# Step 5: Add new features based on contributing factors
df_lion['is_inalchol'] = df_lion['contributing_factor_vehicle_1'].apply(lambda x: 1 if x == "Alcohol Involvement" else 0)
df_lion['is_inunsafespeed'] = df_lion['contributing_factor_vehicle_1'].apply(lambda x: 1 if x == "Unsafe Speed" else 0)
df_lion['is_inbike1'] = df_lion['vehicle_type_code1'].apply(lambda x: 1 if x in ['Bike', 'E-Bike'] else 0)
df_lion['is_inbike2'] = df_lion['vehicle_type_code2'].apply(lambda x: 1 if x in ['Bike', 'E-Bike'] else 0)
df_lion['is_any_inbike'] = df_lion.apply(lambda row: 1 if row['is_inbike1'] == 1 or row['is_inbike2'] == 1 else 0, axis=1)

# Step 6: Fetch historical weather data
daily_weather = fetch_weather_data()

# Step 7: Merge weather data with collision data
daily_weather['borough'] = daily_weather['borough'].astype(int)
df_lion['crash_date'] = pd.to_datetime(df_lion['crash_date'])
daily_weather['date'] = pd.to_datetime(daily_weather['date'])
df_lion = pd.merge(
    df_lion,
    daily_weather,
    left_on=["borough", "crash_date"],
    right_on=["borough", "date"],
    how="left"
)

# Step 8: Save the processed data to the working directory
df_lion.to_csv(location, index=False)

# Step 9: Visualize the processed data
df_lion['StreetWidt'] = df_lion['StreetWidt'].apply(lambda x: 0 if x < 40 else 1)  # Binary street width categorization
df_lion['Number_Tra'] = df_lion['Number_Tra'].apply(lambda x: 3 if x > 2 else x)  # Cap traffic lanes at 3
selected_columns = ['precipitation_sum', 'is_inalchol', 'is_inunsafespeed', 'is_any_inbike', 
                    'StreetWidt', 'Number_Tra', 'Number_Par']

# Calculate proportions for visualization
proportion_data = pd.DataFrame()
for column in selected_columns:
    if column in df_lion.columns:  
        value_counts = df_lion[column].value_counts(normalize=True).sort_index()  
        for value, proportion in value_counts.items():
            proportion_data.loc[value, column] = proportion
proportion_data = proportion_data.fillna(0)

# Plot the stacked bar chart
ax = proportion_data.T.plot(
    kind='bar',
    stacked=True,
    figsize=(12, 8),
    title="Proportion of Variables in 4884 Collisions"
)

for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', label_type='center')

plt.ylabel("Proportion")
plt.xlabel("Variables")
plt.legend(title="Values", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
