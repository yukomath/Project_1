# Salary Prediction Project

This project predicts developer salaries using the Stack Overflow Developer Survey 2025 dataset. It includes data preparation, handling missing values and outliers, one-hot encoding of categorical features, and splitting the data into training and test sets. Three models—Linear Regression, Random Forest, and LightGBM—are trained and evaluated using R², MAE, and RMSE. We analyzed SHAP values, and the models are used to predict salaries for a sample developer profile across different countries.

It applies machine learning techniques learned from the Udacity Data Scientist Nanodegree program, including **Linear Regression** and **Random Forest**, as well as **SHAP analysis** to interpret feature importance. Additionally, **LightGBM** is used as an advanced model explored with the help of ChatGPT.


## Workflow

0. **Preparation** for Googlelab　- Connects Google Drive to the Colab environment, allowing read and write access to files
1. **Data Preparation** — Loads the dataset from Google Drive, selects relevant columns, handles missing values and outliers, performs one-hot encoding for categorical and multi-valued fields, cleans feature names, and finally splits the data into training and test sets for model development.
3. **Build a Salary Prediction Model with Linear Regression** - Builds a salary prediction model using Linear Regression. It trains the model on the prepared features, predicts salaries on the test set, evaluates performance using R², MAE, and RMSE, and then examines the regression coefficients to identify which features have the strongest positive and negative impact on predicted salaries.

4. **Build a Salary Prediction Model with Random Forest** - 
Builds a salary prediction model using Random Forest. It installs and uses the shap library for feature interpretation, trains a tuned Random Forest regressor on the training data, predicts salaries on the test set, and evaluates the model using R², MAE, and RMSE.
Additionally, it performs SHAP analysis to visualize how individual features impact predictions and calculates feature importance to identify the most influential factors on salary, such as country, coding experience, and AI tool usage.

5. **Build a Salary Prediction Model with LightGBM** - Builds a salary prediction model using LightGBM. It installs required packages, trains a gradient boosting model on the training data, predicts salaries on the test set, and evaluates performance with R², MAE, and RMSE.
Additionally, it performs SHAP analysis to interpret feature contributions, visualizes the top 10 most important features, and plots a scatter plot of predicted vs. actual salaries to assess model accuracy.

7. **Review Results** — Reviews and compares the results of all trained models. It creates a summary table of R², MAE, and RMSE for Linear Regression, Random Forest, and LightGBM, and visualizes the comparisons with bar plots.
Additionally, it performs exploratory analysis to examine average salaries by country and programming language, helping to identify patterns and trends in the dataset that may influence salary predictions.

8. **Example case** — Demonstrates how to use the trained models to predict annual salary for a specific profile across different countries. It prepares the input data, applies the same preprocessing and encoding used during training, and generates salary predictions using Linear Regression, Random Forest, and LightGBM.
The results are displayed in a formatted table, rounded to two decimal places, and saved as a PNG image for easy sharing or reporting.


## File Description
project.ipynb : Main notebook 
README.md : Project documentation (this file).

## Author 
Author: Yuko (GitHub: yukomath)

## Acknowledgements
Stack Overflow Annual Developer Survey 2025 dataset: https://survey.stackoverflow.co/

Udacity Data Scientist Nanodegree program 

AI Tool : ChatGPT

## libraries Used 
The following python libraries were used in this project.
pandas==2.2.2
numpy==2.0.2
matplotlib==3.10.0
lightgbm==4.6.0
scikit-learn==1.6.1
seaborn==0.13.2
