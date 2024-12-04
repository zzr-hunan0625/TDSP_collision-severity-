"""
File: lionmap_concat.py
Description:
This script provides a function `lionmap_concat` that processes collision or street data, 
merges it with NYC street-level attributes (from the `lion_name.csv` file), and enriches the 
data with additional features like street width, bike lane presence, speed limits, and more. 

The script utilizes geographic coordinate transformation to match collision points to the 
corresponding streets based on proximity, ensuring accurate attribution of street-related features.

Dependencies:
- pandas
- pyproj
- math
- os

The core functionality includes:
1. Loading and transforming street data (`lion_name.csv`) for feature matching.
2. Applying spatial calculations to find the closest street segment for given latitude/longitude.
3. Enriching the input dataset with street attributes like width, bike lanes, speed limits, etc.

"""

import pandas as pd
from pyproj import Transformer
import math
import os

def lionmap_concat(df1):
    """
    Processes input DataFrame (df1) and merges it with NYC street data for additional features.

    Args:
        df1 (pd.DataFrame): Input DataFrame containing latitude, longitude, and street name.
                            Columns must include 'latitude', 'longitude', 'on_street_name'.

    Returns:
        pd.DataFrame: Processed DataFrame with enriched features like street width, bike lanes, 
                      speed limits, and borough information.
    """
    # Get the current working directory and locate the LION street data file
    current_working_directory = os.getcwd()
    location = os.path.join(current_working_directory, "lion_name.csv")
    
    # Initialize a geographic transformer (WGS84 to EPSG:2263)
    transformer = Transformer.from_crs("epsg:4326", "epsg:2263", always_xy=True)
    
    # Load the LION street data
    df2 = pd.read_csv(location, low_memory=False)
    
    # Standardize and clean street name columns for matching
    df1['on_street_name'] = df1['on_street_name'].str.lower().astype(str)
    df2['Street'] = df2['Street'].str.lower().astype(str)

    def calculate_distance(x1, y1, x2, y2):
        """
        Calculates the Euclidean distance between two points in a 2D space.

        Args:
            x1, y1: Coordinates of the first point.
            x2, y2: Coordinates of the second point.

        Returns:
            float: The distance between the two points.
        """
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def find_corresponding_attributes(lat, lon, street_name, df2):
        """
        Finds the closest street segment and its attributes for a given location and street name.

        Args:
            lat, lon (float): Latitude and longitude of the point.
            street_name (str): Street name to match.
            df2 (pd.DataFrame): LION street data containing segment attributes.

        Returns:
            dict: Dictionary of matched street attributes, defaulting to 2 if no match is found.
        """
        # Filter street data by matching street name
        df_filtered = df2[df2['Street'] == street_name]
        if df_filtered.empty:
            # Return default attributes if no matching street is found
            return {
                'StreetWidt': 0,
                'BikeLane': 0,
                'POSTED_SPE': 0,
                'Number_Tra': 0,
                'Number_Par': 0
            }

        # Transform latitude and longitude to x, y coordinates
        x, y = transformer.transform(lon, lat)

        # Initialize variables for tracking the closest segment
        min_distance = float('inf')
        closest_row = None

        # Iterate through the filtered rows to find the closest segment
        for idx, row in df_filtered.iterrows():
            distance = calculate_distance(x, y, 0.5 * (row['XFrom'] + row['XTo']), 0.5 * (row['YFrom'] + row['YTo']))
            if distance < min_distance:
                min_distance = distance
                closest_row = row

        # Return the attributes of the closest segment, or default values if not found
        if closest_row is not None:
            return {
                'StreetWidt': closest_row['StreetWidt'],
                'BikeLane': closest_row['BikeLane'],
                'POSTED_SPE': closest_row['POSTED_SPE'],
                'Number_Tra': closest_row['Number_Tra'],
                'Number_Par': closest_row['Number_Par'],
                'borough': closest_row['RBoro']
            }

        # Default attributes if no match is found
        return {
            'StreetWidt': 2,
            'BikeLane': 2,
            'POSTED_SPE': 2,
            'Number_Tra': 2,
            'Number_Par': 2,
            'borough': 2
        }

    # Apply the function to find attributes for each row in df1
    df1_attributes = df1.apply(lambda row: find_corresponding_attributes(row['latitude'], row['longitude'], row['on_street_name'], df2), axis=1)
    df1 = df1.join(pd.DataFrame(df1_attributes.tolist(), index=df1.index))

    # Filter out rows with invalid or missing attributes
    df1 = df1[df1['StreetWidt'] != 0]
    df1['POSTED_SPE'] = df1['POSTED_SPE'].fillna(50)
    df1['BikeLane'] = df1['BikeLane'].fillna(0)
    df1['Number_Tra'] = df1['Number_Tra'].fillna(0)
    df1['Number_Par'] = df1['Number_Par'].fillna(0)
    df1['borough'] = df1['borough'].fillna(0)
    df1 = df1[df1['borough'] != 0]

    return df1







