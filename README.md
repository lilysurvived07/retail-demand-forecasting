\# Retail Demand Forecasting



\## Overview

This project builds an end-to-end retail demand forecasting system using Python and machine learning.



The goal is to predict product demand by store and product so retailers can optimize inventory planning and reduce stockouts.



\## Project Pipeline



1\. Generate synthetic retail demand data

2\. Engineer time-series features (lags and rolling averages)

3\. Train a forecasting model

4\. Predict future demand

5\. Visualize results in a Streamlit dashboard



\## Files



generate\_retail\_demand\_data.py  

Creates a synthetic retail sales dataset.



train\_forecast\_model.py  

Trains a Random Forest model for demand forecasting.



forecast\_future.py  

Generates next-day demand predictions.



app.py  

Interactive Streamlit dashboard for exploring forecasts.



\## Outputs



outputs/feature\_importance.png  

Shows which features influence demand most.



outputs/forecast\_plot.png  

Actual vs predicted demand.



outputs/future\_forecast.csv  

Predicted future demand.



outputs/model\_metrics.txt  

Model evaluation metrics.



\## Technologies



Python  

Pandas  

NumPy  

Scikit-Learn  

Matplotlib  

Streamlit



\## Business Value



Retailers can use demand forecasts to:



\- optimize stock levels

\- reduce stockouts

\- improve replenishment planning

\- understand promotion impact



\## Run the Project



Generate dataset



python generate\_retail\_demand\_data.py



Train model



python train\_forecast\_model.py



Generate forecast



python forecast\_future.py



Run dashboard



python -m streamlit run app.py

