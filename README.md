house_prices_kaggle
# Kaggle challenge "House Prices: Advanced Regression Techniques" 

This is the submission of my frist Kaggle challenge: <a href="https://www.kaggle.com/c/house-prices-advanced-regression-techniques">House Prices: Advanced Regression Techniques</a> 

Below, I outline the goal of the challenge as well as how approached building a solution. Please view <a href="https://github.com/HeleneFabia/house-prices-kaggle/blob/master/advanced_regression_house_prices_kaggle.ipynb">my notebook</a> for a more detailed explanation of what I did. 

<p align="center">
  <img width="400" height="200" src="https://cdn.pixabay.com/photo/2019/04/29/20/41/amsterdam-4167026_960_720.png">
</p>

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

## Reflection 

Although I'm quite satisfied with this project and my solution, there are still some things that could be improved:
- My model uses every single feature in the dataframe. Creating new features that bundle some of the given features (e.g. the total number of square feet of a house) could make the model slimmer and thus faster and maybe also more accurate.
- I only experimented with one type of decision tree algorithm, Random Forest Regressor. Maybe ExtraTrees Regressor, Gradient Boosting or XGBoost would yield better results.
- My evaluation could be more detailed and include, for example, an in-depth analysis of those examples with the largest difference between the predicted and the actual prize.

## Problems I faced and how I solved them 

TL;DR: Target mean encoding the categorical features in the validation and test set proved to be difficult, since some categorical feature values appear in the validation and/or test set but not in the training set. Hence, I included an additional parameter in the function encode_dataframe (which I use to target mean encode the training set) (see <a href="https://github.com/HeleneFabia/house-prices-kaggle/blob/master/utils_house_prices.py">utils_house_prices.py</a>). This parameter refers to a dictionary which includes the encoded feature values of each categorical features and the mean value of a categorical feature for those values that are missing in the training set, but appear in the validation and/or test set.

When target mean encoding the categorical features in the validation and test set, I encountered an issue: I want to have the same target mean encoding in all three datasets. However, some categorical feature values appear in the validation and/or test set but not in the training set, so they do not have a target mean that can be mapped onto the feature values in the validation set.

The function encode_dataframe (which I use to target mean encode the training set) (see <a href="https://github.com/HeleneFabia/house-prices-kaggle/blob/master/utils_house_prices.py">utils_house_prices.py</a>) takes the parameter train_set_categorical_encoded_means. This parameter is a dictionary with the column names of the encoded categorical features as keys and the respective target smooth meaning encoding as values. Thus, I now have to build this dictionary which I can then use as an input for the function encode_dataframe to encode the validation set.

First, I create a dictionary (called train_val_categorical_global_means) with the all unique categorical feature values as keys and the global mean of SalePrice as values

    train_val_categorical_global_means = {feature_encoded: 
        {feature_value: df["SalePrice"].mean() for feature_value in df[feature].unique()} 
        for feature, feature_encoded in zip(categorical_features, categorical_features_encoded)}

To explain the above  code further:
- feature_encoded: {...} creates a dictionary key for all column names of the encoded categorical features(e.g. 'Neighborhood_encoded')
- {feature_value: ...} creates dictionary within the above described dictionary, which includes all unique feature values for each categorical feature (e.g. 'Veenker' for the feature 'Neighborhood_encoded') as keys
- df.['SalePrice'].mean() creates a value for each key in the dictionary described above. This values is the global mean of SalePrice (value is always mean of SalePrice)
- The result is a dictionary that looks something like thos: {'Neighborhood_encoded': {'Veenker': 180921.19589041095}}

Second, I create a dictionary (called train_categorical_encoded_means) with every unique categorical feature value (e.g. 'Veenker for the feature 'Neighborhood_encoded') as keys and the respective target mean (as calculated in the training set). This dictionary only includes the encoded features values of the training set. So any feature values present in the validation set, but not in the training set, do not appear in this dictionary.

     train_categorical_encoded_means = {}

     for feature, feature_encoded in zip(categorical_features, categorical_features_encoded):
          # get encoded mean of every categorical feature value and store it in mean_by_var
          mean_by_var = train_set.groupby(feature)[feature_encoded].mean().to_dict()
          # store mean_by_var as the value for every key feature_encoded in a dictionary
          train_categorical_encoded_means[feature_encoded] = mean_by_var

Third, I update the values of the first dictionary I created (train_val_categorical_global_means) with the values of the second dictionary (train_categorical_encoded_means). This means I replace the global mean of SalePrice with the target mean, but this only regards feature values that are present both in the training and validation set. Those feature values that only appear in the validation set remain unchanged (i.e. global mean).

    # loop through keys (encoded_feature) in dict train_val_categorical_global_means
    for encoded_feature in train_val_categorical_global_means.keys():
        # update the value of each key (encoded_feature) with the value of each key in train_categorical_encoded_means
        train_val_categorical_global_means[encoded_feature].update(train_categorical_encoded_means[encoded_feature])
        
The dictionary train_val_categorical_global_means can now be used as a parameter for the function encode_dataframe to encode the validation set.

    val_set = encode_dataframe(val_set, train=False, train_set_categorical_encoded_means=train_val_categorical_global_means)
    
However, this does not solve the problem that some categorical feature values appear in the test set but neither in the training nor validation set, so they do not have a target mean that can be mapped onto the feature values in the test set.

To solve this problem, first, I find out which categorical feature values appear in the test set but not in the training and validation set. To do so, I loop through all categorical features and create two lists (train_set_feature_values, val_set_feature_values and test_set_feature_values) that include the values of every categorical feature of the training, validation, and test set, respectively. I then compare these three lists to each other (looping through every value in test_set_feature_values). If any value is included in the test set but not in the training or validation set, I append both the value and its corresponding feature name to two lists.

    unique_features = []
    unique_feature_values = []

    for feature in categorical_features:
    
        train_set_feature_values = train_set[feature].value_counts().index.tolist()
        val_set_feature_values = val_set[feature].value_counts().index.tolist()
        test_set_feature_values = test_set[feature].value_counts().index.tolist()
    
        train_val_set_feature_values = train_set_feature_values + val_set_feature_values
    
        for value in test_set_feature_values:
            if not value in train_val_set_feature_values:
                unique_features.append(feature)
                unique_feature_values.append(value)

Second, I replace the categorical feature values which only appear in the test set with the global SalePrice mean. To do so, I create a list of those feature names (called unique_features_encoded) that include values which only appear in the test set. I then loop through these feature names, create a new dictionary (called test_categorical_encoding), which includes the unique feature names as keys and an embedded dictionary as values. Within these embedded dictionaries, I store the unqique feature values (as keys) and the global mean of SalePrice (as values). I then use this two-layer dictionary (test_categorical_encoding) to replace the final non-encoded categorical feature values in the test set. 

    unique_features_encoded = [feature + '_encoded' for feature in unique_features]

    for i in range(len(unique_features)):
        test_categorical_encoding = {}
        test_categorical_encoding[unique_features_encoded[i]] = {}
        test_categorical_encoding[unique_features_encoded[i]][unique_feature_values[i]] = train_set.SalePrice.mean()
        
        test_set_selected_features = test_set_selected_features.replace(test_categorical_encoding)
