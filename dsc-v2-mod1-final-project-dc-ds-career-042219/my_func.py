import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import preprocessing

def read_file():
     df = pd.read_csv('~/module1project/dsc-v2-mod1-final-project-dc-ds-career-042219/kc_house_data.csv')
     return df

def heatmap_cols_null(df):
    ''' Plot heatmap for columns with null values '''

    fig, ax = plt.subplots(figsize=(12,4))
    return sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')

def boxplot_func(df, title, x):
    ''' Plot horizontal boxplot for dependent variable price '''

    fig, ax = plt.subplots(figsize=(15,5))
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x, fontsize=14)
    return sns.boxplot(x =x, data = df, orient = 'h', fliersize = 2, width = 0.9, showmeans=True, ax = ax)

def countplot_func(df, title, x):
    ''' Plot countplot for independent variables
    that will be turned into categorical variables '''

    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x, fontsize=14)
    return sns.countplot(x=x, data=df);

def display_null_columns_greater_than_zero(df):
    ''' Return sum of columns with null values '''

    get_sum_per_column = df.isna().sum()
    return get_sum_per_column[get_sum_per_column > 0]


def get_count(df, val):
    ''' Return total number of rows of a column
    if it is equal to val parameter '''

    return len(df[df== val])

def null_values(df):
    ''' Replace view column with null values with 0.0
        drop yr_renovated and id columns
        replace null values in waterfront with 2
        waterfront column with null values'''

    df['view'].fillna(2.0, inplace=True)
    df.drop(['yr_renovated', 'id'], axis=1, inplace=True)
    df['waterfront'].fillna(2, inplace=True)

def dummy_cols(df):
    ''' Return dummies for the following columns:
    waterfront, view, grade, condition, and floors '''

    waterfront = pd.get_dummies(df['waterfront'], prefix='wf')
    view = pd.get_dummies(df['view'], prefix='view')
    grade = pd.get_dummies(df['grade'], prefix='grade')
    condition = pd.get_dummies(df['condition'], prefix='cnd')
    floors = pd.get_dummies(df['floors'], prefix='flr')
    return waterfront, view, grade, condition, floors

def sort_asc_corr(df, value):
    ''' Return correlation in descending order
        based on the specified value parameter
        in our case, we will be using the
        dependent variable price '''

    corr_df = df.corr()[[value]]
    return corr_df.sort_values(by = value, ascending=False)

def assign_median_sqft_basement(df):
    ''' This function filters out null values and '?' from
        sqft_basement column into a temporary dataframeself.
        Calculates the mean of the temporary dataframe and then
        assigns the mean into our original dataframe for the
        sqft_basement column to replace the null values and '?'
        and finally turns it into a float64 dtype '''

    temp_df = df[(~df['sqft_basement'].isna()) & (df['sqft_basement'] != '?')]
    len(temp_df)
    df.loc[(df['sqft_basement'].isna()) | (df['sqft_basement'] == '?'), 'sqft_basement'] = temp_df['sqft_basement'].median()
    df['sqft_basement']= df['sqft_basement'].astype(float)

def get_Xy(df, target, predictor):
    ''' This functions returns the X, y columns to use them
        for the linear regression, cross validation, and ordinary
        least squares models '''

    # c0 = list(range(2,4))
    # c1 = list(range(6,8))
    # c2 = [15]
    # c3 = list(range(17,19))
    # c4 = list(range(20,22))
    # c5 = list(range(26,33))
    # c6 = list(range(37,44))
    # cols = c0+c1+c2+c3+c4+c5+c6
    X1 = df.loc[ : , predictor]
    y1= df.loc[ : , target]
    return X1, y1

def split_train_test(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)
    return X_train, X_test, y_train, y_test

def predict_y(X, y):
    ''' This function splits the X, y into the training and test sets,
        it predicts my y based on the X train set and returns X,y for
        both train and test sets as well as the predicted y '''

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    y_predict = regressor.predict(X_test)
    return X_train, X_test, y_train, y_test, y_predict

def r2_result(y_test, y_predict):
    ''' This function returns the R^2 score '''

    score = r2_score(y_test, y_predict)
    return score

def k_fold(X, y):
    ''' This function performs cross validation on 5, 10, and 20
    folds and returns all three folds results '''

    regressor = LinearRegression()
    regressor = regressor.fit(X, y)
    cv_5  = np.mean(cross_val_score(regressor, X, y, cv=5))
    cv_10 = np.mean(cross_val_score(regressor, X, y, cv=10))
    cv_20 = np.mean(cross_val_score(regressor, X, y, cv=20))
    return cv_5, cv_10, cv_20


def scatter_y(df, y, ncols=3, figsize=(16, 20), wspace=0.2, hspace=0.5, alpha=0.05, color='b'):
    ''' This function scatter plots all our dataframe columns against our
        dependent variable price - The code was provided to us by Jon Solow and Ryan Miller'''

    df_col_list = list(df.columns)
    if (len(df_col_list) % ncols > 0):
        nrows = len(df_col_list)//ncols + 1
    else:
        nrows = len(df_col_list)//ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.subplots_adjust(wspace=0.2, hspace=0.5)

    for i, xcol in enumerate(df_col_list):
        try:
            df.plot(kind='scatter', x=xcol, y=y,
                    ax=axes[i//ncols, i % ncols], label=xcol, alpha=alpha, color=color)
            plt.plot()
        except:
            print('warning: could not graph {} as numbers'.format(xcol))

def sns_context():
    sns.set_context(rc={"axes.titlesize":14,"axes.labelsize":13})

# sns_context()
# f, ax = plt.subplots(figsize=(8,4))
# sns.countplot('month', data=df);
# ax.set_title('Total Houses Sold Per Month')
# sns.despine()

def sns_stripplot(x, y, title, df):
    sns_context()
    f, ax = plt.subplots(figsize=(8,4))
    sns.stripplot(x, y, data=df)
    ax.set_title(title)
    sns.despine()

def sns_barplot(x, y, title, df):
    sns_context()
    f, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x, y, data=df)
    ax.set_title(title)
    sns.despine()

def sns_relplot(x, y, title, df):
    sns_context()
    f, ax = plt.subplots(figsize=(10,5))
    sns.relplot(x, y, data);
    ax.set_title(title)
    sns.despine()

# def scale_data(df, min_max_list, unadj_list):
#
#     min_max_scaler = preprocessing.MinMaxScaler()
#     standard_scaler = preprocessing.StandardScaler()
#
#     X_unadj = pd.DataFrame(columns=unadj_list)
#     X_min_max = pd.DataFrame(columns=min_max_list)
#
#     for col_name in min_max_list:
#         X_min_max[col_name] = df[col_name]
#
#     for col_name in unadj_list:
#         X_unadj[col_name] = df[col_name]
#
#
#     X_min_max = pd.DataFrame(data= min_max_scaler.fit_transform(X_min_max.values), columns=X_min_max.columns)
#
#     scaled_df =pd.concat([X_min_max, X_unadj], axis =1)
#
#     return scaled_df
