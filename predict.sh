#!/bin/bash

# Prediction API endpoint
URL="http://localhost:8000/predict"

# Input JSON
DATA='{
  "median_income": 3.5,
  "housing_median_age": 20.0,
  "avg_rooms": 5.0,
  "avg_bedrooms": 1.0,
  "population": 1000,
  "avg_occupancy": 3.0,
  "latitude": 34.0,
  "longitude": -118.0
}'

# Call API
curl -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "$DATA"
