# Salary Prediction Project

This project predicts developer salaries using the **Stack Overflow Developer Survey 2025** dataset. The project includes comprehensive data preparation, handling missing values and outliers, one-hot encoding of categorical features, and splitting the data into training and test sets. Three regression models—**Linear Regression**, **Random Forest**, and **LightGBM**—are trained and evaluated using **R²**, **MAE**, and **RMSE** metrics. 

Furthermore, we analyzed **SHAP values** to interpret feature importance and used the models to predict salaries for sample developer profiles across different countries, programming languages, education levels, and years of experience.

This project applies machine learning techniques learned from the **Udacity Data Scientist Nanodegree program**, including Linear Regression and Random Forest, along with SHAP analysis for model interpretability. **LightGBM** is employed as an advanced model and explored with the assistance of ChatGPT.

---

## File Description

- **project.ipynb** : Main Jupyter notebook containing data preprocessing, model training, evaluation, SHAP analysis, and salary predictions.  
- **README.md** : Project documentation (this file).
- **requirements.txt : Lists all Python libraries and their versions required to run the project.  
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

## Libraries Used

The following Python libraries were used in this project:

- pandas==2.2.2  
- numpy==2.0.2  
- matplotlib==3.10.0  
- lightgbm==4.6.0  
- scikit-learn==1.6.1  
- seaborn==0.13.2


## Project Workflow

### Section 1: Business Understanding
1. Define the problem: Predict annual developer salaries based on survey data.  
2. Identify key factors affecting salaries: experience, education, programming languages, AI tool usage, and location.  

### Section 2: Data Understanding
1. Load the Stack Overflow 2025 survey dataset.  
2. Select relevant features: `Country`, `EdLevel`, `WorkExp`, `YearsCode`, `LanguageHaveWorkedWith`, `AISelect`, and `ConvertedCompYearly`.  
3. Explore missing values, distributions, and initial patterns in the dataset.  

### Section 3: Data Preparation
1. **Handle Missing Values**  
   - Fill missing categorical values (`Country`, `EdLevel`, `AISelect`) with "Unknown".  
   - Fill missing numerical values (`YearsCode`, `WorkExp`) with the median.  
   - Drop rows with missing target variable (`ConvertedCompYearly`).  

2. **Multi-label Encoding**  
   - Convert `LanguageHaveWorkedWith` (semicolon-separated languages) into multi-hot encoded columns.  

3. **One-hot Encoding for Categorical Columns**  
   - Convert remaining categorical variables to numeric using one-hot encoding.  

4. **Outlier Handling**  
   - Filter `ConvertedCompYearly` to a reasonable range (e.g., 5,000–500,000 USD).  

5. **Clean Column Names**  
   - Replace special characters with underscores for LightGBM compatibility.  

---

### Section 4: Data Modeling
1. **Train/Test Split**  
   - Split dataset into training (80%) and test (20%) sets.  

2. **Linear Regression Model**  
   - Train Linear Regression.  
   - Evaluate with R², MAE, and RMSE.  
   - Examine coefficients to identify influential features.  

3. **Random Forest Model**  
   - Train Random Forest regressor.  
   - Evaluate performance and visualize feature importance using SHAP.  

4. **LightGBM Model**  
   - Train a LightGBM gradient boosting model.  
   - Evaluate performance with R², MAE, and RMSE.  
   - Perform SHAP analysis to interpret feature contributions and visualize top 10 features.  

---

### Section 5: Evaluate the Results

5.1 **Review Results**  
- Compare all models using R², MAE, and RMSE.  
- Visualize comparisons with bar plots.  
- Explore average salaries by country and programming language to detect patterns affecting compensation.  

5.2 **Example Case**  
- Demonstrate salary prediction for a sample developer profile across multiple countries.  
- Apply preprocessing and encoding consistent with training.  
- Generate predictions with all three models and display results in a formatted table.  
- Visualize predictions for different experience levels, education, countries and programming languages.  

---

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
