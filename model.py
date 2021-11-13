"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    feature_vector_df = feature_vector_df
    df_train = feature_vector_df
    df_train['time'] = pd.to_datetime(df_train['time']) #Change from string to datetime

#splitting the time variable into day,month,year and hr
    df_train['time'] = pd.to_datetime(df_train['time'], utc=True, infer_datetime_format=True)
    df_train = df_train.set_index('time')
    median=df_train['Valencia_pressure'].median()
    df_train['Valencia_pressure'].fillna(median,inplace=True)
    Valencia_wind = pd.get_dummies(df_train['Valencia_wind_deg'], drop_first=True)
    df_train =pd.concat([df_train, Valencia_wind], axis = 1)
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([('std_scalar', StandardScaler())])
    for item in list(df_train.columns):
        df_train[item] = [20 for item in df_train[item]]
    df_train['level_10']=df_train['Bilbao_temp_min']
    df_train['level_2']=df_train['Bilbao_temp_min']
    df_train['level_3']=df_train['Bilbao_temp_min']
    for item in list(df_train.columns):
        df_train[item] = [20 for item in df_train[item]]
    #X_train = pipeline.fit_transform(X_train)
    df_train.drop(['Valencia_wind_deg','Seville_pressure'], axis =1, inplace = True)
    #X = df_train.drop('load_shortfall_3h', axis = 1)
    #y =df_train['load_shortfall_3h']
    predict_vector = df_train
    return predict_vector

    # ------------------------------------------------------------------------


    


def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    print (prediction)
    return prediction[0].tolist()

