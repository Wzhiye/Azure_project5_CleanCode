'''
Script for tests on customer churn model code
Author: Zhiye Wen
Date: 11.12.2022
'''

import os
import logging
import churn_library as cls

from math import ceil

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        dataframe = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err

def test_eda():
    '''
    test perform eda function
    '''
    dataframe = cls.import_data("./data/bank_data.csv")
    try:
        cls.perform_eda(dataframe=dataframe)
        logging.info("Testing perform_eda: SUCCESS")
    except KeyError as err:
        logging.error("Testing perform_eda: The %s data is missed", err.args[0])
        raise err
    # check file
    file_list = ["Churn_Histogram", "CustomerAge_Histogram", "Marital_Status_count",
                 "Total_transaction_distribution", "Heatmap"]
    for file in file_list:
        file_name = os.path.join("./images/eda", file+".png")
        try:
            assert os.path.isfile(file_name) is True
            logging.info('File %s was found', file)
        except AssertionError as err:
            logging.error("Testing perform_eda: File %s is not exist", file)
            raise err

def test_encoder_helper():
    '''
    test encoder helper
    '''
    dataframe = cls.import_data("./data/bank_data.csv")
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    #
    try:
        encoded_df = cls.encoder_helper(dataframe=dataframe, category_lst=[], response=None)
        assert encoded_df.equals(dataframe) is True
        logging.info("Testing encoder_helper with void category list: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper with void category list: ERROR")
        raise err

    try:
        encoded_df = cls.encoder_helper(dataframe=dataframe,
                                         category_lst=cat_columns, response=None)
        assert encoded_df.equals(dataframe) is False
        assert encoded_df.columns.equals(dataframe.columns) is True
        logging.info("Testing encoder_helper with void response: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper with void response list: ERROR")
        raise err
    try:
        encoded_df = cls.encoder_helper(dataframe=dataframe,
                                         category_lst=cat_columns, response='Churn')
        assert encoded_df.equals(dataframe) is False
        assert encoded_df.columns.equals(dataframe.columns) is False
        assert (len(encoded_df.columns)==(len(dataframe.columns)+len(cat_columns))) is True
        logging.info("Testing encoder_helper with full category list: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper with full category list: ERROR")
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    dataframe = cls.import_data("./data/bank_data.csv")
    try:
        (_, X_test, _, _) = cls.perform_feature_engineering(
            dataframe=dataframe,
            response='Churn')
        assert 'Churn' in dataframe.columns
        logging.info("Testing perform_feature_engineering: the churn column is in the dataframe")
    except KeyError as err:
        logging.error("Testing perform_feature_engineering: no churn column")
        raise err        
    try:
        assert (X_test.shape[0] ==
               ceil(dataframe.shape[0] *0.3)) is True
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: ERROR")
        raise err

def test_train_models():
    '''
    test train_models
    '''
    dataframe = cls.import_data("./data/bank_data.csv")
    (X_train, X_test, y_train, y_test) = cls.perform_feature_engineering(dataframe=dataframe,
                                                                         response='Churn')
    cls.train_models(X_train, X_test, y_train, y_test)
    models_list = ['logistic_model.pkl', 'rfc_model.pkl']
    for model in models_list:
        try:
            assert os.path.isfile(os.path.join('./models', model)) is True
            logging.info('Testing train_models: File %s is available', model)
        except AssertionError as err:
            logging.error('File %s is not available', model)
            raise err
    images_list = ['rfc_results','lr_results',
                   'feature_importances', 'roc_curve_result']
    for image in images_list:
        try:
            assert os.path.isfile(os.path.join('./images/results', image+'.png')) is True
            logging.info('Testing train_models: File %s is available', image)
        except AssertionError as err:
            logging.error('File %s is not available', image)
            raise err

if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
    