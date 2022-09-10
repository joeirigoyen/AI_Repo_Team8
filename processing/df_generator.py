"""
Equipo: Break the Rules

Integrantes del Equipo:

Eduardo Rodriguez Lopez
Diego Armando Ulibarri Hernandez
Maria Fernanda Ramirez Barragan
Raul Youthan Irigoyen Osorio
Renata Montserrat De Luna Flores
Roberto Valdez Jasso

Nombre: Dataframe Generator

Autor:  Roberto Valdez Jasso

Fecha de Inicio 06/09/2022
Fecha de Finalizacion NAN

Descripcrion breve de codigo:
Este codigo tiene el proposito de generar un dataframe de el csv  Spaceship Titanic  proporcinado
por  Kaggle.

La siguiente clase tendra lo siguientes  proceso:
Info
Corr
Cleaning

"""
import os
import pandas as pd
import numpy as np
import mysql.connector
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

#import servidor
class DataframeGenerator:
    # Initializer of the class
    def __init__(self, source, encoding = "utf_8"):
        """Initializes a dataframe with certain fixes applied before returning it to the user.
        Args:
            csv_file_train (str): the path to the data csv train file
            header (bool, optional): whether the csv contains headers or not. Defaults to None.
            encoding (str, optional): csv file's encoding type. Defaults to "utf_8".
            na_values (str | int | float, optional): character or number used to define a na value within the dataset. Defaults to '?'.
        """
        # Data processing variables
        self.data_folder = os.path.curdir()
        # Dataframe
        self.df = self.get_from_csv(source, encoding)
        self.results = None

    # Connection to MySQL
    def connect_db(self):
        """ cnx = mysql.connector.connect(
        host="db_host",
        user="db_username",
        password="db_password",
        database="db_name",
        port="db_port",
        auth_plugin='mysql_native_password')
        return cnx """
        pass

    # Downloads a dataset from MySQL 
    def get_from_db(self):
        """ cnx = connect_db()
        cur = cnx.cursor()
        cur.execute("SELECT * FROM COLUMN_NAME;)
        query = cur.fetchall()
        COLUMN_NAME = []
        for row in query:
            NAME = {
                "TEST": row[0],
            }
            COLUMN_NAME.append(NAME)"""
        pass

    #Create csv from DataFrame in current folder    
    def create_csv(self, filename):
        filepath = Path(filename)  
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        self.results.to_csv(filepath)

    # Get data from a csv file
    def get_from_csv(self, data, source, encoding):
        # Read csv file from source (must be a valid path within the project)
        data = pd.read_csv(source, encoding=encoding).iloc[:, 1]
        # Send data to processing stage
        data = self.process_dataframe()
        return data

    #Return the percentage of nan values in a column
    def nan_percentage(self, data, column):
        return data[column].isna().sum() * 100 / len(self.df)

    #Imputes the mean, median or mode in the indicated columns depending on the chosen method
    def impute_data(self, data, columns, method='mean'):
        for column in columns:
            #Imputes the mean
            if method == 'mean':
                column_mean = data[column].mean()
                data[column] =data[column].fillna(column_mean)
            #Imputes the median
            elif method == 'median':
                column_median = data[column].median()
                data[column] = data[column].fillna(column_median)
            #Imputes the mode
            elif method == 'mode':
                column_mode = data[column].mode()
                data[column] = data[column].fillna(column_mode)
        return data
        
    #Transform non-numerical labels (as long as they are hashable and comparable) to numerical labels
    def label_encoder(self, data, columns):
        for column in columns:
            labelencoder = LabelEncoder()
            data[column] = labelencoder.fit_transform(data[column])
        return data
    
    def one_hot_encoder():
        pass

    # Process data
    def process_dataframe(self, data: pd.DataFrame, nan_threshold):
        # Remove duplicates from index
        data = data.drop_duplicates(subset=['PassengerId'])
        # Evaluate null values depending on the data types and amount of null values
        nan_columns = data.loc[:, data.isnull().any()].columns
        for column in nan_columns:
            # If percentage of nan values are higher the allowed threshold, drop the column
            if self.nan_percentage(data, column) > nan_threshold:
                data = data.drop(column)
            else:
                # Check which change to df causes the least change in distribution
                pass