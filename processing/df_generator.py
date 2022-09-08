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
# /-----------------------------------------------------------------/
# Librerias
import pandas as pd
import numpy as np
import pylab
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import random


# /-----------------------------------------------------------------/


class DataframeGenerator:
    # Initializer of the class
    # -------------------------------------------------------------------#
    def __init__(self, csv_file_train: str , header: int = 0, encoding: str = "utf_8"):
        """Initializes a dataframe with certain fixes applied before returning it to the user.
        Args:
            csv_file_train (str): the path to the data csv train file
            header (bool, optional): whether the csv contains headers or not. Defaults to None.
            encoding (str, optional): csv file's encoding type. Defaults to "utf_8".
            na_values (str | int | float, optional): character or number used to define a na value within the dataset. Defaults to '?'.
        """
        # Generate dataframe from csv train file
        df_train = pd.read_csv(csv_file_train, header=header, encoding=encoding)
        print(f"File at {csv_file_train} imported successfully as dataframe of shape: {df_train.shape}")
        self.df_original_spaceTitanic = df_train.copy()
        print(f"The  first  ten  rows of the new Dataframe is: \n {self.df_original_spaceTitanic.head(10)}  \n\n and the dataframe shape is : {self.df_original_spaceTitanic.shape}")

    #-------------------------------------------------------------------#
    # Standard Function for the  visualization of the data
    #-------------------------------------------------------------------#

    # Generates the info from the dataframe
    def info(self):
        return  self.df_original_spaceTitanic.info()
    # Checks the Shape of the dataframe
    def shape(self):
        return  self.df_original_spaceTitanic.shape
    # Checks null values of the dataframe
    def null(self):
        return  self.df_original_spaceTitanic.isnull().sum()
    # Checks the amount of rows  of the dataframe
    def check_data(self, amount):
        return  self.df_original_spaceTitanic.head(amount)
    def check_data_porcentage(self):
        return self.df_original_spaceTitanic.isnull().sum() * 100/self.df_original_spaceTitanic.shape[0]
    def data_decribe(self):
        return  self.df_original_spaceTitanic.describe()
    def data_duplicated(self):
        return  self.df_original_spaceTitanic.duplicated().sum()

    # -------------------------------------------------------------------#
    # Standard Function for the Label  of the dataframe

    # Rellenado de valores por frecuencia de  los mismos
    def missing_fill(self, df):
        missing_feature_freq = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
        for feature in missing_feature_freq:
            most_freq = df[feature].value_counts().index[0]
            df[feature] = df[feature].fillna(most_freq)
        return df

    # Extraccion de los dato de la columna  cabin
    # Extraccion de la cabina inicial
    def extractMainDeck(self, s):
        return s.split('/')[0]

    # Extraccion de la del numero de cabina
    def extractNumber(self, s):
        return s.split('/')[1]

    # Extraccion de del area de la cabina (lado de la nave espacial (Ventana o pasillo))
    def extractSide(self, s):
        return s.split('/')[2]

    # Tumbar columnas no Necesarias
    def drop_columns(self, df):
        drop_column = ['PassengerId', 'Cabin', 'Name']  # Duda con el PassengerId
        for ft in drop_column:
            df = df.drop(ft, axis=1)
        return df

    # LabelEncoder
    def encoder(self, df):
        # Columnas para el labeLEncoder
        bool_columns = ['CryoSleep', 'VIP']
        dum_columns = ['Deck']

        # Por cada de las columnas los pasamos a string
        for bool_ft in bool_columns:
            df[bool_ft] = df[bool_ft].astype('str')
        # Por cada de las columnas generamos el label Encoder
        # y de ahi los  pasamos a int
        for dum in dum_columns:
            LabelEnco = LabelEncoder()
            LabelEnco.fit_transform(df[dum])
        df['Num'] = df['Num'].astype('int')
        # gereramos dummies
        df = pd.get_dummies(df)
        return df  # y regresamos el dataframe

    def categoricalPreprossesing(self):
        # Llamada de las funciones
        dataset = self.df_original_spaceTitanic

        df = self.missing_fill(dataset)
        # Llamada de la funcion de los datos de la columna cabina
        df['Deck'] = df['Cabin'].apply(self.extractMainDeck)
        df['Num'] = df['Cabin'].apply(self.extractNumber)
        df['Side'] = df['Cabin'].apply(self.extractSide)
        df = self.drop_columns(df)  # Tumbamos las columnas
        df = self.encoder(df)  # Label Encoder

        return df  # regresamos el df ya finalmente modificado

 
    # -------------------------------------------------------------------#
    # Standard Function for the numeric  and imputation of the dataframe
    # -------------------------------------------------------------------#
    def imputateNumeric(self):
        # Media Room Service
        room_service_mean = self.df_original_spaceTitanic['RoomService'].mean()
        self.df_original_spaceTitanic['RoomService'] = self.df_original_spaceTitanic['RoomService'].fillna(room_service_mean)

        # Age
        age_median = self.df_original_spaceTitanic['Age'].median()
        self.df_original_spaceTitanic['Age'] = self.df_original_spaceTitanic['Age'].fillna(age_median)
        self.df_original_spaceTitanic['Age'].replace({0.0000: age_median}, inplace=True)
        self.df_original_spaceTitanic['Age'] = self.df_original_spaceTitanic['Age'].astype(int)
        # Food Court
        food_court_mean = self.df_original_spaceTitanic['FoodCourt'].mean()
        self.df_original_spaceTitanic['FoodCourt'] = self.df_original_spaceTitanic['FoodCourt'].fillna(food_court_mean)

        # Shopping Mall
        shopping_mall_mean = self.df_original_spaceTitanic['ShoppingMall'].mean()
        self.df_original_spaceTitanic['ShoppingMall'] = self.df_original_spaceTitanic['ShoppingMall'].fillna(shopping_mall_mean)

        # SPA
        spa_mean = self.df_original_spaceTitanic['Spa'].mean()
        self.df_original_spaceTitanic['Spa'] = self.df_original_spaceTitanic['Spa'].fillna(spa_mean)

        # VRdeck
        vrdeck_mean = self.df_original_spaceTitanic['VRDeck'].mean()
        self.df_original_spaceTitanic['VRDeck'] = self.df_original_spaceTitanic['VRDeck'].fillna(vrdeck_mean)

    # -------------------------------------------------------------------#


# Call sample
address1 = r'C:\Users\rober\OneDrive\Escritorio\Repositorios Git\AI_Repo_Team8\pre-data\Titanic\train.csv' # First csv(training)
prueba = DataframeGenerator(address1) # class generations

print('/------------------------------------/')
# Info
print(prueba.info())
print('/------------------------------------/')
# shape
print(prueba.shape())
print('/------------------------------------/')
# Is Null basic
print(prueba.null())
print('/------------------------------------/')
# check_data
print(prueba.check_data(10))
print('/------------------------------------/')
# check_data_porcentage
print(prueba.check_data_porcentage())
print('/------------------------------------/')
# data_decribe
print(prueba.data_decribe())
print('/------------------------------------/')
# data_duplicated
print(prueba.data_duplicated())
print('/------------------------------------/')
# Standard Label Encoder format
prueba.categoricalPreprossesing()

print('/------------------------------------/')
print('/------------------------------------/')
# Standard Numeric Imputation format
prueba.imputateNumeric()
print(prueba.null()) # We check that has  zeros missing values
                     # and we see that there is no missing values
print('/------------------------------------/')
