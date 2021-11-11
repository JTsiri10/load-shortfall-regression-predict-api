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
    
    # ------------------------------------------------------------------------

    feature_vector_df['time'] = pd.to_datetime(feature_vector_df['time']) #Change from string to datetime

#splitting the time variable into day,month,year and hr
    feature_vector_df['Time_day']=feature_vector_df['time'].dt.day

    feature_vector_df['Time_month']=feature_vector_df['time'].dt.month

    feature_vector_df['Time_hour']=feature_vector_df['time'].dt.hour

    feature_vector_df['Time_year']=feature_vector_df['time'].dt.year

    df_time=feature_vector_df.filter(regex='Time', axis=1)


    target_variable=feature_vector_df['load_shortfall_3h']
    df_time = pd.concat([df_time,target_variable], axis = 1)

    feature_vector_df["sp"]=feature_vector_df["Seville_pressure"].str.replace("sp", "")
    feature_vector_df['sp'] = pd.to_numeric(feature_vector_df['sp'])

    #drop old column and rename new column
    feature_vector_df.drop(["Seville_pressure"],axis=1,inplace=True)
    #rename new column and view new table
    feature_vector_df.rename({'sp': 'Seville_pressure'}, axis=1, inplace=True)

    #converting Valencia_wind_deg to Vwd
    #removing the sp in the seville pressure column
    feature_vector_df["Vwd"]=feature_vector_df["Valencia_wind_deg"].str.replace("level_", "")
    feature_vector_df['Vwd'] = pd.to_numeric(feature_vector_df['Vwd'])

    #drop old column and rename new column
    feature_vector_df.drop(["Valencia_wind_deg"],axis=1,inplace=True)

    #rename new column and view new table
    feature_vector_df.rename({'Vwd': 'Valencia_wind_deg'}, axis=1, inplace=True)
    #Dropping columns
    feature_vector_df = feature_vector_df.drop(['time', 'Barcelona_temp_min','Barcelona_temp_max','Bilbao_temp_max','Madrid_temp_min','Madrid_temp_max','Seville_temp_min','Valencia_temp_min'], axis = 1)
    return feature_vector_df



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
    return prediction[0].tolist()
