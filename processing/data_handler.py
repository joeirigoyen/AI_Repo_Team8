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
    def __init__(self, csv_path, encoding = "utf_8", nan_threshold=20.0, id_index=0):
        """Initializes a dataframe with certain fixes applied before returning it to the user.
        Args:
            csv_file_train (str): the path to the data csv train file
            header (bool, optional): whether the csv contains headers or not. Defaults to None.
            encoding (str, optional): csv file's encoding type. Defaults to "utf_8".
            na_values (str | int | float, optional): character or number used to define a na value within the dataset. Defaults to '?'.
        """
        # Data cleaning variables
        self.nan_threshold = nan_threshold
        self.ordinal_columns = ['CryoSleep', 'VIP', 'Transported', 'Deck', 'Side']
        self.non_ordinal_columns = ['HomePlanet', 'Destination']
        # Initialize data
        self.df = pd.read_csv(csv_path, encoding=encoding)
        self.df = self.process_data(self.df, id_index)

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
    def clean_data(self, data, id_column, nan_threshold):
        # Remove duplicates from id column
        data = data.drop_duplicates(subset=[id_column])
        # Delete columns with a higher percentage than the allowed missing values threshold
        if nan_threshold > 1:
            nan_threshold /= 100
        data = data.dropna(thresh=data.shape[0] * nan_threshold, how='all', axis=1)
        # Perform imputation on numerical attributes with nans
        num_data = data.select_dtypes(include=np.number)
        data= self.impute_num_data(data, num_data.columns[num_data.isna().any()].tolist())
        # Impute categorical data using most frequent values
        cat_cols= data.select_dtypes(exclude=np.number).columns
        data = self.impute_cat_data(data, cat_cols)
        return data

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
    def encode_cat_data(self, data, label_cols, one_hot_cols, sample=False):
        # Perform label encoding
        if sample:
            if 'Transported' in label_cols:
                label_cols.remove('Transported')
        data = self.label_encode(data, label_cols)
        # Perform one hot encoding
        data = self.one_hot_encode(data, one_hot_cols)
        data.drop(one_hot_cols, axis=1, inplace=True)
        return data

    def engineer_data(self, data):
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
        data['InGroup'] = data['Group'].apply(lambda x : 1 if x in groups else 0)
        data.drop(['PassengerId', 'Group'], axis=1, inplace=True)
        # Make TotalSpent column by getting the sum of all services per passenger
        service_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        data['TotalSpent'] = data[service_columns].sum(axis=1)
        data.drop(service_columns, axis=1, inplace=True)
        # Use ranges to categorize service columns
        data['TotalSpent'] = data['TotalSpent'].apply(lambda x : 0 if 0 <= x < 750 else 1 if 750 <= x < 1200 else 2 if 1200 <= x < 2500 else 3)
        # Categorize age column
        data['Age'] = data['Age'].apply(lambda x : 0 if 0 <= x < 2 else 1 if 2 <= x < 5 else 2 if 5 <= x < 13 else 3 if 13 <= x < 20 else 4 if 20 <= x < 40 else 5 if 40 <= x < 60 else 6)
        # Remove unnecessary columns
        data.drop('PassengerNumber', axis=1, inplace=True)
        return data

    # Process original data before uploading to database
    def process_data(self, data, id_index, sample=False):
        # Check data type
        if isinstance(data, str):
            data = pd.read_csv(data)
        # Get id column
        id_column = data.columns[id_index]
        # Clean data
        data.drop('Name', axis=1, inplace=True)
        data = self.clean_data(data, id_column, self.nan_threshold)
        # Perform feature engineering
        data = self.engineer_data(data)
        data = self.encode_cat_data(data, self.ordinal_columns, self.non_ordinal_columns, sample=sample)
        # Move results column to the rightmost position
        if not sample:
            temp_column = data['Transported']
            data.drop('Transported', axis=1, inplace=True)
            data.insert(data.shape[1], 'Transported', temp_column)
        return data
