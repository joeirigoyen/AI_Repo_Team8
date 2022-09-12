"""
Team: Break the Rules

Team Members:   Eduardo Rodriguez Lopez
                Diego Armando Ulibarri Hernandez
                Maria Fernanda Ramirez Barragan
                Raul Youthan Irigoyen Osorio
                Renata Montserrat De Luna Flores
                Roberto Valdez Jasso

Class name: Data Handler

Authors:    Raúl Youthan Irigoyen Osorio
            Eduardo Rodriguez López
            María Fernanda Ramirez Barragán
            Roberto Valdez Jasso

Creation Date: September 7th, 2022

Process dataset, split it into training and testing sets, perform cross-validation and process small samples from users.
"""
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

#import servidor
class DataHandler:
    # Initializer of the class
    def __init__(self, source, encoding = "utf_8"):
        """Initializes a dataframe with certain fixes applied before returning it to the user.
        Args:
            csv_file_train (str): the path to the data csv train file
            header (bool, optional): whether the csv contains headers or not. Defaults to None.
            encoding (str, optional): csv file's encoding type. Defaults to "utf_8".
            na_values (str | int | float, optional): character or number used to define a na value within the dataset. Defaults to '?'.
        """
        # Data cleaning variables
        self.nan_threshold = 20.0
        # Initialize data
        self.source = source
        self.df = pd.read_csv(source, encoding=encoding)
        self.id_column = self.df.columns[0]
        self.process_dataframe()

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

    def compute_smd(self, data_1, data_2):
        data_1 = np.array(data_1)
        data_2 = np.array(data_2)
        return np.abs((data_1.mean() - data_2.mean())) / np.sqrt((np.power(data_1.std(), 2) + np.power(data_2.std(), 2)) / 2)

    #Imputes the mean, median or mode in the indicated columns depending on the chosen method
    def impute_num_data(self, data, columns, signif_level=0.05):
        for column in columns:
            # Original values
            techniques = [data[column].mean(), data[column].median(), data[column].mode()[0]]
            tech_names = ['mean', 'median', 'mode']
            # Use the technique with the highest p-value
            max_p_value = signif_level
            max_index = -1
            for i in range(len(techniques)):
                temp_data = data[column].fillna(techniques[i], inplace=False)
                curr_p = ks_2samp(data[column], temp_data).pvalue
                if curr_p >= max_p_value:
                    max_index = i
            if max_index < 0:
                max_index = len(techniques) - 1
            data[column] = data[column].fillna(techniques[max_index], inplace=False)
        return data

    # Impute categorical features
    def impute_cat_data(self, data, columns):
        # Impute data using most frequent value
        cat_imputer = SimpleImputer(strategy="most_frequent")
        result = cat_imputer.fit_transform(data[columns])
        # Rearrange result into dataframe
        result = pd.DataFrame(result, columns=columns)
        data[result.columns] = result
        return data
    
    # Standardize current dataframe
    def normalize_dataframe(self, data, exclude=[]):
        scaler = MinMaxScaler()
        num_cols = data.select_dtypes(include=np.number).columns
        for column in exclude:
            if column in num_cols:
                index = num_cols.get_loc(column)
                num_cols = num_cols.delete(index)
        data[num_cols] = scaler.fit_transform(data[num_cols])
        return data

    # Process the original dataframe
    def clean_dataframe(self, id_column, nan_threshold):
        # Remove duplicates from id column
        self.df = self.df.drop_duplicates(subset=[id_column])
        # Delete columns with a higher percentage than the allowed missing values threshold
        if nan_threshold > 1:
            nan_threshold /= 100
        self.df = self.df.dropna(thresh=self.df.shape[0] * nan_threshold, how='all', axis=1)
        # Perform imputation on numerical attributes with nans
        num_data = self.df.select_dtypes(include=np.number)
        self.df= self.impute_num_data(self.df, num_data.columns[num_data.isna().any()].tolist())
        # Impute categorical data using most frequent values
        cat_cols= self.df.select_dtypes(exclude=np.number).columns
        self.df = self.impute_cat_data(self.df, cat_cols)
        # Scale and normalize data
        self.df = self.normalize_dataframe(self.df, exclude=['Age'])

    #Transform ordinal attributes (as long as they are hashable and comparable) into numerical labels
    def label_encode(self, data, columns):
        for column in columns:
            labelencoder = LabelEncoder()
            data[column] = labelencoder.fit_transform(data[column])
            data[column] = pd.to_numeric(data[column])
        return data

    # Create dummy columns for non-ordinal attributes
    def one_hot_encode(self, data, columns):
        for column in columns:
            new_columns = pd.get_dummies(data[column])
            for new_column in new_columns.columns:
                data[new_column] = new_columns[new_column]
        return data

    # Encode the data's categorical attributes
    def encode_cat_data(self, data, label_cols, one_hot_cols):
        # Perform label encoding
        data = self.label_encode(data, label_cols)
        # Perform one hot encoding
        data = self.one_hot_encode(data, one_hot_cols)
        data.drop(one_hot_cols, axis=1, inplace=True)
        return data

    def engineer_data(self, data: pd.DataFrame):
        # Split cabin into parts and delete original column
        split_cabin_data = data['Cabin'].str.split('/', n=2, expand=True)
        data['Deck'] = split_cabin_data[0]
        data['Room'] = pd.to_numeric(split_cabin_data[1])
        data['Side'] = split_cabin_data[2]
        data.drop('Cabin', axis=1, inplace=True)
        # Split passenger ids into groups and passenger numbers
        split_id_data = data['PassengerId'].str.split('_', n=1, expand=True)
        data['Group'] = split_id_data[0]
        data['PassengerNumber'] = split_id_data[1]
        # Make column to determine if passengers are in groups or not
        groups = set(data[data.groupby('Group')['Group'].transform('size') > 1]['Group'])
        data['InGroup'] = data['Group'].apply(lambda x : x in groups)
        data.drop(['PassengerId', 'Group'], axis=1, inplace=True)
        return data

    def process_dataframe(self):
        # Clean data
        self.df.drop('Name', axis=1, inplace=True)
        self.clean_dataframe(self.id_column, self.nan_threshold)
        # Perform feature engineering
        self.df = self.engineer_data(self.df)
        self.df = self.encode_cat_data(self.df, ['CryoSleep', 'VIP', 'Transported', 'Deck', 'Side', 'InGroup'], ['HomePlanet', 'Destination'])
        # Move results column to the rightmost position
        temp_column = self.df['Transported']
        self.df.drop('Transported', axis=1, inplace=True)
        self.df.insert(self.df.shape[1], 'Transported', temp_column)

    def process_sample(self, data, id_column):
        data.drop('Name', axis=1, inplace=True)
        data = self.clean_sample(data, id_column, self.nan_threshold)
        data = self.engineer_data(self.df)
        data = self.encode_cat_data(self.df, ['CryoSleep', 'VIP', 'Transported', 'Deck', 'Side', 'InGroup'], ['HomePlanet', 'Destination'])
        return data
