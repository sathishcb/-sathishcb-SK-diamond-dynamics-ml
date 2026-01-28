💎 Diamond Dynamics: Price Prediction & Market Segmentation

An end-to-end Machine Learning + Deep Learning project that predicts diamond prices and segments diamonds into market categories using clustering. The solution is deployed as an interactive Streamlit web app.

📌 Problem Statement

Diamond pricing depends on multiple quality attributes such as carat, cut, color, clarity, and dimensions.
This project builds ML models to:

✔ Predict diamond price
✔ Segment diamonds into market groups
✔ Provide an interactive tool for pricing and classification

🎯 Objectives

Build regression models to predict diamond price

Build an ANN model for comparison

Perform market segmentation using K-Means clustering

Use PCA for cluster visualization

Deploy everything using Streamlit

🧠 Skills Demonstrated

Data Cleaning & Preprocessing

Exploratory Data Analysis (EDA)

Feature Engineering

Outlier & Skewness Handling

Feature Selection

Machine Learning Regression

Artificial Neural Networks (ANN)

K-Means Clustering

PCA (Dimensionality Reduction)

Streamlit Web App Deployment

📊 Dataset Information

Rows: 53,940

Features: 10

Source: Diamonds Dataset

Key Columns
Feature	Description
carat	Weight of diamond
cut	Cut quality (Fair → Ideal)
color	Color grade (D best → J worst)
clarity	Inclusion grade
x, y, z	Dimensions in mm
price	Price in USD (converted to INR)
🧹 Data Preprocessing

Removed invalid dimension values (0 in x, y, z)

Handled missing values

Converted price from USD → INR

Removed outliers using IQR method

Checked skewness in numerical features

📈 Exploratory Data Analysis

Price distribution plots

Price vs Cut, Color, Clarity

Carat vs Price relationship

Correlation heatmap

Pairwise feature relationships

🧩 Feature Engineering

New features created:

Volume = x × y × z

Price per Carat

Dimension Ratio

Carat Category (Light / Medium / Heavy)

🎯 Feature Selection

Used Random Forest Feature Importance to identify the most impactful features for price prediction.

🤖 Regression Models Used
Model	Purpose
Linear Regression	Baseline model
Decision Tree	Non-linear modeling
Random Forest	Best performing model
KNN	Instance-based learning
XGBoost	Gradient boosting
ANN (Neural Network)	Deep learning comparison

Evaluation Metrics: MAE, RMSE, R² Score

🧩 Clustering (Market Segmentation)

Algorithm: K-Means

Features scaled using StandardScaler

Optimal clusters chosen using Elbow Method

PCA used for 2D cluster visualization

Cluster Labels
Cluster Name	Description
💍 Affordable Small Diamonds	Low carat, budget stones
✨ Mid-range Balanced Diamonds	Moderate size & price
👑 Premium Heavy Diamonds	High carat, luxury stones
🌐 Streamlit Web Application
Features

✔ Price Prediction
✔ Market Segment Prediction
✔ Preset Diamond Profiles
✔ Downloadable Prediction Report
✔ Clean & Interactive UI

▶ Run the App
streamlit run app.py

📁 Project Structure
diamond-dynamics/
│
├── data/
│   └── diamonds.csv
│
├── models/
│   ├── price_model.pkl
│   ├── encoder.pkl
│   ├── cluster_model.pkl
│   ├── scaler.pkl
│   └── cluster_names.json
│
├── notebook/
│   └── Diamond_Dynamics.ipynb
│
└── app.py

🚀 Real-World Applications

Dynamic pricing for diamond retailers

Inventory segmentation

Luxury recommendation systems

Customer targeting & marketing

🛠 Tech Stack

Python

Pandas, NumPy

Scikit-learn

XGBoost

TensorFlow / Keras

Matplotlib, Seaborn

Streamlit

👤 Author

Your Name Here
Machine Learning Enthusiast 💎
