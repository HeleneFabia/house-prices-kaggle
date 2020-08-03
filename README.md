house_prices_kaggle
# Project No. 2: Kaggle competition "House Prices: Advanced Regression Techniques" 

This is the submission of my frist Kaggle competition: <a href="https://www.kaggle.com/c/house-prices-advanced-regression-techniques">House Prices: Advanced Regression Techniques</a> 

Below, I outline the goal of the competition as well as how approached building a solution. Please view <a href="https://github.com/HeleneFabia/house-prices-kaggle/blob/master/advanced_regression_house_prices_kaggle.ipynb">my notebook</a> for a more detailed explanation of what I did. 

<h2> Dataset </h2>

The 1460x82 dataset provided by Kaggle contains the data of 1460 houses in Ames, Iowa. Each house is described by 79 features, including general information such as the size in square feet and the year it was sold, as well as very specific details such as the slope of the property or the roof material (For the complete list of features, please view data_description.txt)

The goal of this competition is to build a model that that predicts the sale price of each house in the test set.

<h2> Data Preprocessing </h2>

To preprocess the data, I handle categorical, ordinal and numerical features separately. For categorical feature, I fill any missing data with the string "missing" and target mean encode them. For ordinal features, I map any strings with the according numerical value (e.g. Excellent: 5, Good: 4, Regular: 3, etc.) and fill any any missing values with the mode of the respective feature. For numerical features, I fill any  missing values with the mean of the respective feature.

<h2> Modelling </h2>

I used to different models: Random Forest Regressors and Extra Trees Regressor. For both, I use RandomSearch and GridSearch to find good hyperparameters.

<h2> Evaluation </h2>

For both models, I look at the root mean squared error to evaluate them as well at feature importance in order to gain a better understanding of the model's output.

RMSE of Random Forest Regressor:

RMSE of Extra Trees Regressor:

Based on the evalutaion of the predictions of the validation set, I use ... as he model to predict house prices in the test set. This gives me a RMSE of ... on the public leaderboard and ... on the private leaderboard.

<h2> Reflection </h2>

Although I'm quite satisfied with this project and my solution, there are still some things that could be improved:

My model uses every single feature in the dataframe. Creating new features that bundle some of the given features (e.g. the total number of square feet of a house) could make the model slimmer and thus faster and maybe more accurate (since fewer features may reduce the risk of overfitting).

I only experimented with two types of decision tree algorithm, Random Forest Regressor and Extra Trees Regressor. Maybe Gradient Boosting or XGBoost would yield better results.
