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
        # Initialize data
        self.source = source
        self.df = pd.read_csv(source, encoding=encoding)
        self.id_column = self.df.columns[0]
        # Clean data
        self.nan_threshold = 20.0
        self.df = self.clean_data(self.df, self.id_column, self.nan_threshold)
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

    def compute_smd(self, data_1, data_2):
        data_1 = np.array(data_1)
        data_2 = np.array(data_2)
        return np.abs((data_1.mean() - data_2.mean())) / np.sqrt((np.power(data_1.std(), 2) + np.power(data_2.std(), 2)) / 2)

    #Imputes the mean, median or mode in the indicated columns depending on the chosen method
    def impute_data(self, data, columns):
        return data

    # Process data
    def clean_data(self, data, id_column, nan_threshold):
        # Remove duplicates from id column
        data = data.drop_duplicates(subset=[id_column])
        # Delete columns with a higher percentage than the missing values threshold
        if nan_threshold > 1:
            nan_threshold /= 100
        data = data.dropna(thresh=data.shape[0] * nan_threshold, how='all', axis=1)
        # Evaluate feasible imputation techniques for the rest of the columns with missing data
        nan_columns = data.loc[:, data.isnull().any()].columns
        
        return data

    #Transform non-numerical labels (as long as they are hashable and comparable) to numerical labels
    def label_encoder(self, data, columns):
        for column in columns:
            labelencoder = LabelEncoder()
            data[column] = labelencoder.fit_transform(data[column])
        return data
    
    def one_hot_encoder():
        pass