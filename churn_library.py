'''
Script for gather the useful function
Author: Zhiye Wen
Date: 11.12.2022
'''

# import libraries
import os
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# path library
SAVE_PATH = {
    'image_path': './images/eda',
    'data_path': './data/',
    'results_path': './images/results',
    'models_path': './models'
}


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    dataframe = pd.read_csv(pth)
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return dataframe


def perform_eda(dataframe):
    '''
    perform eda on df and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    '''
    # churn histogram
    plt.figure(figsize=(20, 10))
    dataframe['Churn'].hist()
    plt.savefig(
        fname=os.path.join(
            SAVE_PATH['image_path'],
            'Churn_Histogram.png'))
    plt.close()

    # Customer_Age histogram
    plt.figure(figsize=(20, 10))
    dataframe['Customer_Age'].hist()
    plt.savefig(
        fname=os.path.join(
            SAVE_PATH['image_path'],
            'CustomerAge_Histogram.png'))
    plt.close()

    # Matrial_Status
    plt.figure(figsize=(20, 10))
    dataframe.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(
        fname=os.path.join(
            SAVE_PATH['image_path'],
            'Marital_Status_count.png'))
    plt.close()

    # 'Total_Trans_Ct'
    plt.figure(figsize=(20, 10))
    sns.histplot(dataframe['Total_Trans_Ct'], kde=True)
    plt.savefig(
        fname=os.path.join(
            SAVE_PATH['image_path'],
            'Total_transaction_distribution.png'))
    plt.close()

    # Heat map
    plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(fname=os.path.join(SAVE_PATH['image_path'], 'Heatmap.png'))
    plt.close()


def encoder_helper(dataframe, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument
            that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    encoded_df = dataframe.copy(deep=True)

    for category in category_lst:
        cate_column = []
        category_groups = dataframe.groupby(category).mean()['Churn']

        for val in dataframe[category]:
            cate_column.append(category_groups.loc[val])

        cat_name = category + '_' + response if response else category
        encoded_df[cat_name] = cate_column

    return encoded_df


def perform_feature_engineering(dataframe, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional
              argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    # encoding dataframe
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    encoded_df = encoder_helper(
        dataframe=dataframe,
        category_lst=cat_columns,
        response=response)

    # get X and y
    X = pd.DataFrame()
    X[keep_cols] = encoded_df[keep_cols]

    y = encoded_df['Churn']

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # random forest
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_test, y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(
        fname=os.path.join(
            SAVE_PATH['results_path'],
            'rfc_results.png'))
    plt.close()

    # Logistic Regression
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_test, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(
        fname=os.path.join(
            SAVE_PATH['results_path'],
            'lr_results.png'))
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Feature importances
    importances = model.best_estimator_.feature_importances_

    # Sort Feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(25, 15))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save the image
    plt.savefig(fname=os.path.join(output_pth, 'feature_importances.png'))
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # init model
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # grid search
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # fit model
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    # get results
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # save the best model
    joblib.dump(
        cv_rfc.best_estimator_,
        os.path.join(
            SAVE_PATH['models_path'],
            'rfc_model.pkl'))
    joblib.dump(
        lrc,
        os.path.join(
            SAVE_PATH['models_path'],
            'logistic_model.pkl'))

    # Compute ROC curve
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    rfc_plot = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=axis, alpha=0.8)
    lrc_plot = plot_roc_curve(lrc, X_test, y_test, ax=axis, alpha=0.8)
    plt.savefig(
        fname=os.path.join(
            SAVE_PATH['results_path'],
            'roc_curve_result.png'))
    plt.close()

    # feature importances
    feature_importance_plot(
        model=cv_rfc,
        X_data=X_test,
        output_pth=SAVE_PATH['results_path'])
    # classification report
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)


if __name__ == '__main__':
    # Import data
    DF = import_data(pth=os.path.join(SAVE_PATH['data_path'], 'bank_data.csv'))

    # Perform EDA
    perform_eda(dataframe=DF)

    # Feature engineering
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        dataframe=DF, response='Churn')

    # Model training,prediction and evaluation
    train_models(X_train=X_TRAIN,
                 X_test=X_TEST,
                 y_train=Y_TRAIN,
                 y_test=Y_TEST)
