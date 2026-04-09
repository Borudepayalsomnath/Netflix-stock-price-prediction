# Netflix Stock Price Prediction 📈

## Project Overview
This project predicts **Netflix (NFLX) stock prices** using Machine Learning models.  
The dataset contains historical stock data such as **Date, Open, Close, and Volume**.

The project performs:
- Data loading
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Model training
- Model evaluation
- Future stock price prediction (60 days)
- Data visualization

## Dataset
The dataset used is **NFLX.csv**, which contains historical Netflix stock price data.

Example columns:
- Date
- Open
- High
- Low
- Close
- Volume

## Technologies Used
- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Machine Learning Models Used
The project compares multiple models:

1. **Linear Regression**
2. **Random Forest Regressor**
3. **Decision Tree Regressor**
4. **Logistic Regression** (for price direction prediction)

## Project Workflow

1. Import required libraries  
2. Load dataset  
3. Perform Exploratory Data Analysis (EDA)  
4. Data preprocessing  
5. Feature engineering (Year, Month, Day)  
6. Train-test split  
7. Feature scaling using StandardScaler  
8. Train machine learning models  
9. Evaluate model performance  
10. Predict future stock prices (60 days)  
11. Visualize results

## Evaluation Metrics

Regression models:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

Classification model:
- Accuracy
- Classification Report

## Visualizations

The project includes:
- Historical stock price trend
- Actual vs predicted prices
- Correlation heatmap
- Scatter plot comparison

## How to Run the Project

1. Clone or download the project
2. Install required libraries
3. Place the dataset `NFLX.csv` in the project folder
4. Open the notebook

Run:

```bash
jupyter notebook