# TDSP_collision severity_analysis
·Project: NYC Collision Data Analysis

·Description: This project analyzes New York City collision data by integrating weather data, street attributes, and other relevant features. It employs machine learning models to predict incident severity and visualize variable distributions. 

·The project is organized into four Python scripts:
1.fetch_weather_data.py: Fetches historical weather data for NYC boroughs.
2.lionmap_concat.py: Enriches collision data with street-level attributes from the LION map.
3.pre-processing.py: Cleans and integrates collision, street, and weather data for analysis.
4.main.py: Trains machine learning models and evaluates their performance in predicting collision severity.
Project Workflow

·Data Collection:
1.Fetch collision data from NYC Open Data API.
2.Fetch weather data using the fetch_weather_data.py script.
3.Use LION map attributes to enrich the dataset with street-level information.

·Data Preprocessing:
1.Clean and filter raw data to remove missing or invalid entries.
2.Merge datasets to form a comprehensive analysis-ready dataset.
3.Label incidents by severity and engineer features based on contributing factors.

·Model Training:
1.Train machine learning models: K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Random Forest, and Decision Tree.
2.Handle class imbalance using SMOTE (Synthetic Minority Oversampling Technique).
3.Evaluate models using cross-validation and performance metrics like precision, recall, and F1-score.

·Visualization:
1.Generate bar charts showing variable proportions in collisions.
2.Visualize model performance using confusion matrices and classification metrics.

·Python Scripts
1. fetch_weather_data.py
Purpose: Fetch historical daily precipitation data for NYC boroughs.
Inputs:
Start and end dates for the data.
Threshold for precipitation classification.
Outputs: A DataFrame containing precipitation data (date, precipitation_sum, and borough).
2. lionmap_concat.py
Purpose: Merge collision data with street attributes from the LION map.
Functionality:
Transforms geographic coordinates to match street segments.
Matches collision data to the closest street segment and enriches it with attributes like street width, bike lanes, and speed limits.
3. pre-processing.py
Purpose: Clean and integrate collision, weather, and street data into a single dataset.
Steps:
Fetch and clean collision data from the NYC Open Data API.
Merge street and weather data with collision data.
Label incidents based on severity (no injury, injury, fatality).
Engineer features for machine learning models.
4. main.py
Purpose: Train and evaluate machine learning models.
Steps:
Standardize features and handle class imbalance.
Train models (KNN, SVM, Random Forest, Decision Tree).
Evaluate models using cross-validation.
Visualize results with charts and confusion matrices.
