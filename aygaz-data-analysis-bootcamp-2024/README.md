# Adult Census Income Prediction - Data Analysis Project

Github: https://github.com/alperugurca/aygaz-data-analysis-bootcamp-2024

Kaggle: https://www.kaggle.com/code/alperugurca/adult-income-dataset

## Project Summary
This project presents a comprehensive data analysis of the Adult Census Income dataset from the UCI Machine Learning Repository. The primary goal is to predict whether an individual's annual income exceeds $50,000 based on various demographic and socioeconomic factors from US census data.

## Dataset Overview
- **Source**: [UCI Machine Learning Repository](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)
- **Sample Size**: 32,561 records
- **Features**: 14 independent variables
- **Target Variable**: Income (>$50K, ≤$50K)

## Feature Description
### 1. Demographic Features
- **Age**: Individual's age
- **Race**: Racial category
- **Gender**: Gender identity
- **Native Country**: Country of origin

### 2. Education and Employment
- **Education**: Highest education level achieved
- **Education-Num**: Years of formal education
- **Occupation**: Job category
- **Workclass**: Type of employment
- **Hours-per-Week**: Weekly working hours

### 3. Additional Features
- **Marital Status**: Current marital status
- **Relationship**: Family role/relationship status
- **Capital Gain**: Investment gains
- **Capital Loss**: Investment losses
- **FNLWGT**: Final weight (census sampling weight)

## Project Objectives
1. Conduct detailed exploratory data analysis
2. Examine inter-variable relationships
3. Implement data cleaning and preprocessing
4. Perform statistical analysis with visualizations
5. Identify optimal machine learning models for income prediction

## Technical Approach
- Data manipulation: Pandas and NumPy
- Visualization: Matplotlib and Seaborn
- Feature engineering: Scikit-learn

## Key Findings
Our analysis revealed:
1. Strong correlation between education level and income
2. Positive relationship between working hours and earnings
3. Significant impact of age and work experience on income levels

## Applications
This analysis can be valuable for:
- HR departments developing salary policies
- Career counseling services providing income forecasts
- Organizations establishing recruitment frameworks

## Recommended Machine Learning Models
1. Random Forest Classifier
2. Gradient Boosting Classifier
3. XGBoost

These models were selected for their:
- Ability to handle mixed data types (categorical and numerical)
- Capacity to capture non-linear relationships
- Proven performance in similar classification tasks
