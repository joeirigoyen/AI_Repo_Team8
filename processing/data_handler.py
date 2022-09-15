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
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SMOTE

#import servidor
class DataHandler:
    # Initializer of the class
    def __init__(self, csv_path, encoding = "utf_8", nan_threshold=20.0, id_index=0, result_label='Transported'):
        """Initializes a dataframe with certain fixes applied before returning it to the user.
        Args:
            csv_file_train (str): the path to the data csv train file
            header (bool, optional): whether the csv contains headers or not. Defaults to None.
            encoding (str, optional): csv file's encoding type. Defaults to "utf_8".
            na_values (str | int | float, optional): character or number used to define a na value within the dataset. Defaults to '?'.
        """
        # Data cleaning variables
        self.nan_threshold = nan_threshold
        self.ordinal_columns = ['Transported']
        self.non_ordinal_columns = ['VIP', 'Deck', 'CryoSleep', 'HomePlanet', 'Destination', 'Side']
        # Data engineering variables
        self.relevant_columns = []
        self.result_label = result_label
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
            if self.result_label in label_cols:
                label_cols.remove(self.result_label)
        data = self.label_encode(data, label_cols)
        # Perform one hot encoding
        data = self.one_hot_encode(data, one_hot_cols)
        data.drop(one_hot_cols, axis=1, inplace=True)
        return data

    # Alter columns so that they can be interpreted more easily by the models (dataset specific)
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
        # Determine how many passengers are in each group
        groups = pd.Series(data['Group'].unique()).values
        members = pd.Series(data.groupby('Group')['Group'].value_counts().values).values
        group_dict = {}
        for group, num in zip(groups, members):
            group_dict[group] = num
        data['GroupMembers'] = data['Group'].apply(lambda g : group_dict[g])
        data.drop(['PassengerId', 'Group'], axis=1, inplace=True)
        # Make TotalSpent column by getting the sum of all services per passenger
        service_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        data['TotalSpent'] = data[service_columns].sum(axis=1)
        # Remove unnecessary columns
        data.drop(['PassengerNumber', 'Name'], axis=1, inplace=True)
        return data

    # Perform oversampling to reduce minority class omission
    def oversample_data(self, data, result_label):
        smote = SMOTE(random_state=42)
        x = data.drop(result_label, axis=1)
        y = data[result_label]
        resampled_x, resampled_y = smote.fit_resample(x, y)
        data = pd.concat([resampled_x, resampled_y], axis=1)
        return data

    # Select the n most correlated features
    def select_features(self, data: pd.DataFrame, result_label, top_features=10, sample=False):
        if not sample:
            top_corr_indices = data.corr().abs().nlargest(top_features + 1, result_label)[result_label].index
            top_corr_indices = top_corr_indices.drop(result_label)
            self.relevant_columns = top_corr_indices
            data = pd.concat([data[self.relevant_columns], data[result_label]], axis=1)
        else:
            data = data[self.relevant_columns]
        return data

    # Process original data before uploading to database
    def process_data(self, data, id_index, sample=False):
        # Check data type
        if isinstance(data, str):
            data = pd.read_csv(data)
        # Get id column
        id_column = data.columns[id_index]
        # Clean data
        data = self.clean_data(data, id_column, self.nan_threshold)
        # Perform feature engineering
        for column in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
            data.loc[data['CryoSleep'] == True, column] = 0.0
        data['CryoSleep'] = data['CryoSleep'].apply(lambda val: "CryoSlept" if val else "NoCryoSleep")
        data['VIP'] = data['VIP'].apply(lambda val: "IsVIP" if val else "NotVIP")
        data = self.engineer_data(data)
        data = self.encode_cat_data(data, self.ordinal_columns, self.non_ordinal_columns, sample=sample)
        # Move Transported column to the rightmost position
        if not sample:
            temp_column = data[self.result_label]
            data.drop(self.result_label, axis=1, inplace=True)
            data.insert(data.shape[1], self.result_label, temp_column)
            # Process outliers
            """ factor = LocalOutlierFactor(contamination=0.025)
            outliers = factor.fit_predict(data)
            data = data[np.where(outliers == 1, True, False)] """
            # Oversample data
            data = self.oversample_data(data, self.result_label)
        # Get the most correlated columns
        data = self.select_features(data, self.result_label, sample=sample, top_features=15)
        return data
