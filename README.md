# Employment Trend Analysis and Anomaly Detection

## Project Overview
This repository contains the code and resources for the **AI-driven Employment Trend Analysis and Anomaly Detection** project (CMPE 255, Option 1). We analyze California employment data to forecast future trends and automatically detect anomalous shifts.

## Dataset
- **Source:** California Open Data Portal, Current Employment Statistics (CES)  
  https://data.ca.gov/dataset/current-employment-statistics-ces-2/resource/98b69522-557e-464a-a2be-4226df433da1  
- **Contents:** Monthly employment counts by industry (1990–present), both seasonally adjusted and raw.

## Repository Structure

- `assets/`  
  - Vector SVG figures for the report  
- `dashboard/`  
  - Streamlit/Dash application code  
- `EDA.ipynb`  
  - Exploratory Data Analysis & Anomaly Detection notebook  
- `Analysis.ipynb`  
  - Employment trend analysis using ARIMA, Prophet, LSTM  
- `requirements.txt`  
  - Python dependencies  
- `README.md`  
  - This file 

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/veenasjsu/cmpe255project.git
   cd cmpe255project

2. **install dependencies**
pip install -r requirements.txt


3. **jupyter notebook notebooks/EDA.ipynb**
-Cleans and visualizes raw CES data.
-Outputs vector-format SVG figures in assets/.
-Forecasting & Anomaly Detection


4. **jupyter notebook notebooks/Analysis.ipynb**
-Runs ARIMA, Prophet, and LSTM models.
-Evaluates with MAE/RMSE.
-Interactive Dashboard

5. **Dash**
-Launches a web app at http://localhost:8050 (Anomaly detection) or http://127.0.0.1:8052 (Trend analysis).
-Choose industry, adjust forecast horizon and contamination rate, view results.

6. **Contributors & Task Distribution**
Veena Vyshnavi Garre
Data acquisition, cleaning, and preprocessing
Exploratory Data Analysis and visualizations
Time series forecasting: ARIMA and Prophet, LSTM model implementation
Model evaluation and comparison
Dashboard development

7. **Requirements**
Python 3.8+

Packages:
pandas, numpy, scikit-learn, statsmodels, prophet, torch, plotly, Dash
