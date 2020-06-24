# LISTS AND DICTIONARIES 

# list of all categorical features in the dataset
categorical_features = ['MSSubClass', 'MSZoning',  'Street', 'Alley', 'LotShape', 'LandContour', 
'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 
'Condition1', 'Condition2', 'BldgType', 'HouseStyle',  'RoofStyle', 
'RoofMatl', 'Exterior1st', 'Exterior2nd', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 
'Foundation', 'Heating', 'Electrical', 'Functional', 
'GarageType', 'Fence', 'MiscFeature',
'MoSold', 'SaleType', 'SaleCondition']

# list of all categorical features in the dataset that are target mean encoded
categorical_features_encoded = [feature + '_encoded' for feature in categorical_features] 

# list of all ordinal features in the dataset
ordinal_features = ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                    'CentralAir', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageFinish',  
                    'GarageCond', 'PavedDrive', 'PoolQC']

# list of all ordinal features in the dataset that are target mean encoded
ordinal_features_encoded = [feature + '_encoded' for feature in ordinal_features]

# dictionary that assigns numbers to ordinal feature values 
ordinal_encoding = {
     'ExterQual_encoded': 
     {'Ex':5, 'Gd':4, 'TA': 3, 'Fa':2, 'Po': 1},
     'ExterCond_encoded':
     {'Ex':5, 'Gd':4, 'TA': 3, 'Fa':2, 'Po': 1},
     'BsmtQual_encoded':
     {'Ex':5, 'Gd':4, 'TA': 3, 'Fa':2, 'Po': 1},
     'BsmtCond_encoded':
     {'Ex':5, 'Gd':4, 'TA': 3, 'Fa':2, 'Po': 1},
     'BsmtExposure_encoded':
     {'Gd':4, 'Av':3, 'Mn':2, 'No': 1},
     'CentralAir_encoded':
     {'Y': 1, 'N': 0},
     'HeatingQC_encoded':
     {'Ex':5, 'Gd':4, 'TA': 3, 'Fa':2, 'Po': 1},
     'KitchenQual_encoded':
     {'Ex':5, 'Gd':4, 'TA': 3, 'Fa':2, 'Po': 1},
     'FireplaceQu_encoded':
     {'Ex':5, 'Gd':4, 'TA': 3, 'Fa':2, 'Po': 1},
     'GarageQual_encoded':
     {'Ex':5, 'Gd':4, 'TA': 3, 'Fa':2, 'Po': 1},
     'GarageFinish_encoded':
     {'Fin':3, 'RFn':2, 'Unf':1},
     'GarageCond_encoded':
     {'Ex':5, 'Gd':4, 'TA': 3, 'Fa':2, 'Po': 1},
     'PavedDrive_encoded':
     {'Y':3, 'P':2, 'N':1},
     'PoolQC_encoded':
     {'Ex':4, 'Gd':3, 'TA': 2, 'Fa':1}}

# list of all numerical features in the dataset 
numerical_features = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea','BsmtFinSF1', 
'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
'2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',  'BsmtHalfBath',
'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
'Fireplaces', 'GarageYrBlt', 'GarageCars', 
'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
'3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold']

# FUNCTIONS 

def calc_smooth_mean(df, by, on, m):
    """ 
    cal_smooth_mean takes a data frame df and target mean encodes a feature 
    Parameters:
    -----------
    df: the dataframe
    by: the name of the response variable (name of the feature that you want to target mean encode)
    on: the name of the target variable
    m: the smoothing constant
    Returns:
    --------
    the target mean encoded response variable
    """
    mean = df[on].mean()
    #print(f'MEAN:{mean}')
    agg = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    #print(f'COUNTS:{counts}')
    means = agg['mean']
    #print(f'MEANS:{means}')
    smooth = (counts * means + m * mean) / (counts + m)
    #print(f'SMOOTH:{smooth}')
    return df[by].map(smooth)

def encode_dataframe(df, train=True, train_set_categorical_encoded_means=None):
    """ 
    encode_dataframe takes a data frame df and performs a series of changes on it:
    - it fills any missing categorical values with the string 'missing'
    - it replaces ordinal values with a numerical mapping
    - it fills any missing ordinal values with the mode of the respective feature
    - it fills any missing numerical values with the mean of the respective feature
    - all categorical features are target smooth mean encoded
    Parameters:
    -----------
    df: the data frame that you want to process
    train: True if the data frame is the training set, False if it is the validation or test set
    train_set_categorical_encoded_means: a dictionary of the name of the encoded categorical features as keys and the respective target smooth meaning encoding as values; only necessary if df is the validation or test set and train=False; default = None
    Returns:
    --------
    df: the processed dataframe
    """
    
    # Processing of categorical features
    
    missing_categorical_features = df[categorical_features].isna().any()
    missing_categorical_features = missing_categorical_features[missing_categorical_features==True].index.tolist()
    df.fillna(value = {feature: 'missing' for feature in missing_categorical_features}, inplace = True)
    
    if train:
        ## train set
        for i in range(len(categorical_features)):
            df[categorical_features_encoded[i]] = calc_smooth_mean(df, categorical_features[i], 'SalePrice', 10)
    else:
        ## val/test set
        for i in range(len(categorical_features)):
            df[categorical_features_encoded[i]] = df[categorical_features[i]].copy()

        df = df.replace(train_set_categorical_encoded_means)
        
    # Processing of ordinal features
    
    for i in range(len(ordinal_features)):
        df[ordinal_features_encoded[i]] = df[ordinal_features[i]].copy()    
        
    df = df.replace(ordinal_encoding)
    
    missing_ordinal_features = df[ordinal_features_encoded].isna().any()
    missing_ordinal_features = missing_ordinal_features[missing_ordinal_features==True].index.tolist()
    df.fillna(value = {feature: df[feature].mode()[1] for feature in missing_ordinal_features}, inplace = True)
    
    # Processing of numerical features
    
    missing_numerical_features = df[numerical_features].isna().any()
    missing_numerical_features = missing_numerical_features[missing_numerical_features==True].index.tolist()
    df.fillna(value = {feature: df[feature].mean() for feature in missing_numerical_features}, inplace = True)
        
    return df