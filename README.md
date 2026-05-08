# 🚂 IBB Rail System — Station-Based Passenger Demand Forecasting

A machine learning and deep learning research project for predicting 
station-based passenger demand in Istanbul's rail transportation system.

## 📌 Overview

This project compares 4 machine learning and 4 deep learning algorithms 
for time-series passenger demand forecasting. The best-performing models 
(Random Forest + CatBoost) were combined into a hybrid ensemble architecture 
using gradient-based weight optimization. This repository contains the 
implementation of the hybrid model.

## 🤖 Models

**Machine Learning:** Random Forest, XGBoost, LightGBM, CatBoost  
**Deep Learning:** LSTM, CNN-1D, RNN, GRU  
**Hybrid:** RF + CatBoost Ensemble (Gradient-Based Weight Optimization)

> ⚠️ This repository contains only the hybrid RF+CatBoost ensemble model.
> Other models are not included in this repository.

## 🌍 Environmental Features

In addition to time-series data, the following environmental factors 
were incorporated into model training:

- 🌤️ Weather conditions
- 📅 Weekday / Weekend
- 🎒 School days (secondary education)
- 🏛️ Public holidays
- 🕌 Religious holidays

## 🏆 Results

Random Forest and CatBoost outperformed deep learning models, 
likely due to the seasonal patterns and high number of stations 
in the dataset. A hybrid RF+CatBoost ensemble was developed as 
the final model.

## 🚀 Live Demo

👉 [Try the Streamlit App](https://ibb-rail-system-ensmble-model-prediction.streamlit.app/)

## 📄 Academic Work

This project is the basis of a graduation thesis.
Academic paper pending publication (ETSC 2026).

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
