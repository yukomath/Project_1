# Salary Prediction Project

This project analyzes the Stack Overflow Annual Developer Survey 2025 dataset to predict **annual developer salaries** (`ConvertedCompYearly`). 

By predicting developer salaries and analyzing feature importance, the project helps to understand which factors — such as experience, education level, country, employment type, and programming languages — most influence salary.  

The project also provides insights into salary trends across different **countries** and **programming languages**, making it a practical application of data science concepts and model interpretation.

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
   
