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
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DataframeGenerator:
    # Initializer of the class
    # -------------------------------------------------------------------#
    def __init__(self, source: str , header: int = 0, encoding: str = "utf_8"):
        """Initializes a dataframe with certain fixes applied before returning it to the user.
        Args:
            csv_file_train (str): the path to the data csv train file
            header (bool, optional): whether the csv contains headers or not. Defaults to None.
            encoding (str, optional): csv file's encoding type. Defaults to "utf_8".
            na_values (str | int | float, optional): character or number used to define a na value within the dataset. Defaults to '?'.
        """
        # Get data from source
        self.df = self.get_from_source(source, header, encoding)