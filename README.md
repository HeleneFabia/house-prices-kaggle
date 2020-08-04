house_prices_kaggle
# Project No. 2: Kaggle challenge "House Prices: Advanced Regression Techniques" 

This is the submission of my frist Kaggle challenge: <a href="https://www.kaggle.com/c/house-prices-advanced-regression-techniques">House Prices: Advanced Regression Techniques</a> 

Below, I outline the goal of the challenge as well as how approached building a solution. Please view <a href="https://github.com/HeleneFabia/house-prices-kaggle/blob/master/advanced_regression_house_prices_kaggle.ipynb">my notebook</a> for a more detailed explanation of what I did. 

## Dataset 

The 1460x82 dataset provided by Kaggle contains the data of 1460 houses in Ames, Iowa. Each house is described by 79 features, including general information such as the size in square feet and the year it was sold, as well as very specific details such as the slope of the property or the roof material (For the complete list of features, please view <a href="https://github.com/HeleneFabia/house-prices-kaggle/blob/master/data_description.txt">data_description.txt</a>) 

The goal of this competition is to build a model that that predicts the sale price of each house in the test set.

## Data Preprocessing 

To preprocess the data, I handle categorical, ordinal and numerical features separately. For categorical feature, I fill any missing data with the string "missing" and target mean encode them. For ordinal features, I map any strings with the according numerical value (e.g. Excellent: 5, Good: 4, Regular: 3, etc.) and fill any any missing values with the mode of the respective feature. For numerical features, I fill any  missing values with the mean of the respective feature.

## Modelling

I used Random Forest Regressor as well as RandomSearch and GridSearch to find good hyperparameters for the model:

    model = RandomForestRegressor(
        bootstrap = True, 
        max_depth = 60, 
        max_features = 'auto', 
        min_samples_leaf = 1, 
        min_samples_split = 4, 
        n_estimators = 1000 , 
        n_jobs = -1
        )

## Evaluation 

To evaluate my model, I have a look at the first 15 predictions to gain a rough idea of how well my model did:

![Prediction vs Actual Prices](https://github.com/HeleneFabia/house-prices-kaggle/blob/master/evaluation.png)

I look at at the feature importance in order to gain a better understanding of the model's output as well as at the root mean squared logarithmic error, which serves as the evaluation metric of this challenge

Feature importance:

![Feature Importance](https://github.com/HeleneFabia/house-prices-kaggle/blob/master/feature_importance.png)

RMSLE of the predictions of the validation set: 
0.13463

RMSLE of the <a href="https://github.com/HeleneFabia/house-prices-kaggle/blob/master/house_prices_sub.csv">predictions of the test set</a> (as shown on the Kaggle leaderboard):
0.14693

## Problems I faced and how I solved them 
- target mean encoding: 
- 

## Reflection 

Although I'm quite satisfied with this project and my solution, there are still some things that could be improved:
- My model uses every single feature in the dataframe. Creating new features that bundle some of the given features (e.g. the total number of square feet of a house) could make the model slimmer and thus faster and maybe also more accurate.
- I only experimented with one type of decision tree algorithm, Random Forest Regressor. Maybe ExtraTrees Regressor, Gradient Boosting or XGBoost would yield better results.
