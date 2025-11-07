# Salary Prediction Project

This project analyzes the Stack Overflow Annual Developer Survey 2025 dataset to predict **annual developer salaries** (`ConvertedCompYearly`). 

By predicting developer salaries and analyzing feature importance, the project helps to understand which factors — such as experience, education level, country, employment type, and programming languages — most influence salary.  

The project also provides insights into salary trends across different **countries** and **programming languages**, making it a practical application of data science concepts and model interpretation.

It applies machine learning techniques learned from the Udacity Data Scientist Nanodegree program, including **Linear Regression** and **Random Forest**, as well as **SHAP analysis** to interpret feature importance. Additionally, **LightGBM** is used as an advanced model explored with the help of ChatGPT.


## Overview

This project analyzes the **Stack Overflow Annual Developer Survey 2025** dataset to predict **annual developer salaries** (`ConvertedCompYearly`).

The notebook walks through a complete data science workflow:
1. **Data Cleaning and Preparation** — Missing values are handled, and categorical features such as education level, country, and employment type are processed.
2. **Exploratory Data Analysis (EDA)** — Average salaries are visualized by country and by programming language to understand global trends.
3. **Feature Engineering** — Numerical and categorical variables are transformed to create meaningful input features for modeling.
4. **Model Training and Evaluation** — Several models are trained to predict developer salaries:
   - **Linear Regression** (baseline model)
   - **Random Forest** (non-linear ensemble model)
   - **LightGBM** (gradient boosting model for improved accuracy and speed)
5. **Model Interpretation (Explainability)** — Using **SHAP (SHapley Additive exPlanations)**, the notebook visualizes which features most strongly affect salary predictions.
6. **Results Visualization** — Predicted vs actual salaries are plotted, and the importance of top 10 factors influencing salaries is summarized.


## 1. Installation

This project is designed to run on **Google Colab**.  

### Steps to Set Up:

1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. Open the project notebook directly from GitHub:
   - Click **File → Open notebook → GitHub**.
   - Paste the repository URL: `https://github.com/yukomath/Project_1`.
   - Select `project.ipynb`.
3. Install required Python packages by running the following cell:

```python
!pip install lightgbm shap seaborn
Most other packages such as pandas, numpy, and scikit-learn are already included in Colab.

2. Project
This project analyzes the Stack Overflow Annual Developer Survey 2025 dataset to predict annual developer salaries (ConvertedCompYearly).
It applies machine learning techniques learned from the Udacity Data Scientist Nanodegree program, including Linear Regression and Random Forest, as well as SHAP analysis to interpret feature importance.
Additionally, LightGBM is used as an advanced model explored with the help of ChatGPT.
The project also provides insights into salary trends across different countries and programming languages, making it a practical application of data science concepts and model interpretation.

3️⃣ Motivation
The purpose of this project is to apply the concepts learned in the Udacity Data Scientist Nanodegree program to a real-world dataset.
By predicting developer salaries and analyzing feature importance, the project helps to understand which factors — such as experience, education level, country, employment type, and programming languages — most influence salary.
It also provides a practical opportunity to explore advanced models like LightGBM and interpret model predictions using SHAP values, enhancing both technical skills and data-driven insights.

4️⃣ File Description
project.ipynb : Main notebook containing data cleaning, model training, evaluation, SHAP analysis, and visualizations.
data/ : Folder containing the raw Stack Overflow survey dataset (CSV files). (if applicable)
README.md : Project documentation (this file).
requirements.txt (optional) : List of Python packages required to run the notebook.
5️⃣ How to Interact with Your Project
Open Google Colab.
Open the notebook directly from GitHub:
File → Open notebook → GitHub → Paste repository URL → Select project.ipynb
Install required packages (if needed):
!pip install lightgbm shap seaborn
Run the notebook cells step by step:
Data Cleaning: Handle missing values and prepare features.
Model Training: Train Linear Regression, Random Forest, and LightGBM models.
Model Evaluation: Check R², MAE, RMSE for each model.
Feature Importance Analysis: Use SHAP to interpret model predictions.
Visualizations: Compare predicted vs actual salaries and explore salary trends by country and programming language.
Optionally, modify parameters or features to experiment with different models or analyses.
6️⃣ Licensing & Authors
License: MIT License
Author: Yuko (GitHub: yukomath)
Contact: (optional) your email or GitHub profile link
7️⃣ Acknowledgements
Stack Overflow Annual Developer Survey 2025 dataset: https://survey.stackoverflow.co/
SHAP documentation: https://shap.readthedocs.io/
Udacity Data Scientist Nanodegree program for foundational ML concepts
Open-source Python libraries: pandas, numpy, scikit-learn, LightGBM, matplotlib, seaborn





5. Licensing, Authors and Acknowledgements

URL
https://survey.stackoverflow.co/2024/
   
