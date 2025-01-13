# 17K Mobile Strategy Games Analysis

## 1. Project Overview
This project analyzes a dataset of 17,007 strategy games from the Apple App Store, focusing on predicting user ratings by using the first letter of the game name.

## 2. Data Source
[17K Apple App Store Strategy Games](https://www.kaggle.com/datasets/tristan581/17k-apple-app-store-strategy-games/)

## 3. Key Features
- Feature: Implemented one hot encoding first letter of games
- Target: Average User Ratings
- Implementation of supervised and unsupervised learning models

## 4. Models Used
### 4.1. Supervised Learning
- Linear Regression, Ridge, Lasso
- Decision Tree, Random Forest, Gradient Boosting
- SVR, KNN

### 4.2. Unsupervised Learning
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Model

## 5. Key Findings

1. Limited Predictive Power: The first letter of a game's name has minimal predictive power for its user rating, as evidenced by low R-squared scores across supervised learning models.

2. Clustering Insights: Unsupervised learning algorithms revealed some patterns, but clusters were not highly distinct, suggesting weak natural groupings based on the first letter and user rating.

3. Feature Limitations: The analysis was constrained by using only the first letter and user rating, likely contributing to the modest performance of both predictive models and clustering algorithms.

## 6. How to Run
1. Clone the repository
2. Install required libraries: `pip install -r requirements.txt`
3. Run the Jupyter notebook: `jupyter notebook 17K-Mobile-Strategy-Games.ipynb`

## 7. Result
The experiment shows us that using only the first letters of game titles and applying a one-hot encoder to predict user ratings is likely an overly simplistic approach. It may not capture enough meaningful information from the data, as the first letters of game titles do not necessarily contain relevant features for predicting user preferences or ratings.
