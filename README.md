# Salary Prediction Project

This project predicts developer salaries using the **Stack Overflow Developer Survey 2025** dataset. The project includes comprehensive data preparation, handling missing values and outliers, one-hot encoding of categorical features, and splitting the data into training and test sets. Three regression models—**Linear Regression**, **Random Forest**, and **LightGBM**—are trained and evaluated using **R²**, **MAE**, and **RMSE** metrics. 

Furthermore, we analyzed **SHAP values** to interpret feature importance and used the models to predict salaries for sample developer profiles across different countries, programming languages, education levels, and years of experience.

This project applies machine learning techniques learned from the **Udacity Data Scientist Nanodegree program**, including Linear Regression and Random Forest, along with SHAP analysis for model interpretability. **LightGBM** is employed as an advanced model and explored with the assistance of ChatGPT.

---

## File Description

- **project.ipynb** : Main Jupyter notebook containing data preprocessing, model training, evaluation, SHAP analysis, and salary predictions.  
- **README.md** : Project documentation (this file).
- **requirements.txt** : Lists all Python libraries and their versions required to run the project.  
  This file allows anyone to install the exact dependencies using `pip install -r requirements.txt`.

---

## Author

- **Author:** Yuko  
- **GitHub:** [yukomath](https://github.com/yukomath)

---

## Acknowledgements

- **Stack Overflow Annual Developer Survey 2025 dataset**: [https://survey.stackoverflow.co/](https://survey.stackoverflow.co/)  
- **Udacity Data Scientist Nanodegree program**  
- **AI Tool:** ChatGPT

---

## Environment
Developed in Google Colab using Python 3.12.12

---

## Libraries Used

The following Python libraries were used in this project:

- pandas==2.2.2  
- numpy==2.0.2  
- matplotlib==3.10.0  
- lightgbm==4.6.0  
- scikit-learn==1.6.1  
- seaborn==0.13.2

---

## Requirements

To install dependencies:
pip install -r requirements.txt

---

## Usage
Open `project.ipynb` in Google Colab and run all cells sequentially.


---

## Project Workflow

### Section 1: Business Understanding
- Brief Description:
Analyze the 2025 Stack Overflow Survey to understand and predict developer salaries.

- Business Objective:
Identify key salary-driving factors and support job seekers in making informed career decisions.

- Analytic Goals:
Build predictive models and use SHAP to explain which features influence salary outcomes.

- Key Questions addressed in this project:
1. What salary can job seekers expect based on their experience?
2. How do different programming languages or technologies impact salary expectations?
3. What is the average salary for job seekers in different countries or regions?
4. How does education level affect salary potential for job seekers?


### Section 2: Data Understanding
### 2.1 Data Source

Uses the Stack Overflow Developer Survey 2025 dataset, which provides global information about developers’ demographics, skills, and salaries.

### 2.2 Data Loading and Initial Exploration

Loads the dataset in Google Colab and checks its size, structure, and preview to understand initial patterns.

### 2.3 Selected Variables for Analysis

Focuses on key factors affecting salary—country, education, experience, languages, AI usage, and annual compensation.

### 2.4 Data Structure and Variable Types

The dataset contains numerical, categorical, and multi-label fields requiring specific preprocessing such as encoding and cleaning.

### 2.5 Data Quality Issues

Identifies missing values, inconsistent numeric formats, multi-label fields, and extreme salary outliers that must be addressed.

### 2.6 Initial Findings

Initial exploration reveals geographic diversity, common programming languages, salary distribution patterns, and experience trends.

### Section 3: Data Preparation
### 3.1 Handling Missing Values

Categorical missing values are filled with "Unknown" and numerical fields with medians; rows missing salary are removed.

### 3.2 Multi-label Encoding

Programming languages in LanguageHaveWorkedWith are converted into multi-hot encoded binary columns.

### 3.3 One-hot Encoding for Categorical Columns

Remaining categorical variables (Country, EdLevel, AISelect) are converted to numeric using one-hot encoding.

### 3.4 Outlier Handling

Salary values are filtered to a reasonable range (USD 5,000–500,000) to remove extreme outliers.

### 3.5 Column Name Cleaning

Special characters are replaced with underscores to ensure compatibility with LightGBM.

### Section 4: Data Modeling
### 4.1 Train/Test Split

The dataset is split into 80% training and 20% testing, preparing features and the salary target for model training.

### 4.2 Model Training (LR, RF, LightGBM)

Three models—Linear Regression, Random Forest, and LightGBM—are trained to capture linear, nonlinear, and boosted patterns.

### 4.3 Model Evaluation

Models are evaluated using R², MAE, and RMSE, with LightGBM achieving the best accuracy across all metrics.

### Section 5: Evaluate the Results

### 5.1 Salary Expectations by Experience

Experience is one of the strongest predictors of salary; earnings grow rapidly in early years and plateau after mid-career.

### 5.2 Impact of Programming Languages

Different languages affect salary differently; high-demand languages add value, though their impact depends on the full skill profile.

### 5.3 Salary Differences Across Countries

Geographic location strongly influences compensation; the U.S. and Canada show higher predicted salaries than Japan or Italy for the same skill set.

### 5.4 Effect of Education Level

Higher education offers some salary advantage, but its influence decreases as work experience accumulates, making practical skills more important.









