import pandas as pd
import numpy as np
import pylab
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import random

"""
Idea y si juntamos andos set

"""



# Lectura de csv
address1 = r'C:\Users\rober\OneDrive\Escritorio\Repositorios Git\AI_Repo_Team8\pre-data\Titanic\train.csv' # First csv(training)
address2 = r'C:\Users\rober\OneDrive\Escritorio\Repositorios Git\AI_Repo_Team8\pre-data\Titanic\test.csv'  # Second csv(test)
header: int = 0
encoding: str = "utf_8"
# Generate dataframe from csv train file
df_train = pd.read_csv(address1, header=header, encoding=encoding)
print(f"File at {address1} imported successfully as dataframe of shape: {df_train.shape}")
# Generate dataframe from csv test file
df_test = pd.read_csv(address2, header=header, encoding=encoding)
print(f"File at {address2} imported successfully as dataframe of shape: {df_test.shape}")
# We put both data sets together to use them in your analysis and cleaning later
df_original_spaceTitanic = pd.concat([df_train, df_test], axis=1, ignore_index = True  ,sort = False)
df_original_spaceTitanic.columns = ['PassengerId',	'HomePlanet', 'CryoSleep', 'Cabin', 'Destination',	'Age',	'VIP',	'RoomService',	'FoodCourt',	'ShoppingMall',	'Spa',	'VRDeck',	'Name',	'Transported']
print(f"The  first  ten  rows of the new Dataframe is: \n {df_original_spaceTitanic.head(10)}  \n\n and the dataframe shape is : {df_original_spaceTitanic.shape}")
print(df_original_spaceTitanic.duplicated().sum())





