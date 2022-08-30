import os
import pandas as pd


class DataframeGenerator:
    def __init__(self, directory: str, encoding="utf-8", na_values='?'):
        # Initialize file list
        self.files = []
        # Get list of files in directory
        for file in os.listdir(directory):
            self.files.append(os.path.abspath(os.path.join(directory, file)))
        # Let user decide which set to use for training and testing
        self.train, self.test = self.choose_sets()
        self.train = self.clean_data(pd.read_csv(self.train, encoding=encoding, na_values=na_values))
        self.test = self.clean_data(pd.read_csv(self.test, encoding=encoding, na_values=na_values))

    def clean_data(self, df):
        # For now, just remove the na values from the data
        return df.dropna()

    def display_files(self):
        print(f"Files in data directory:")
        for index, file in enumerate(self.files):
            print(f"{index + 1}) {os.path.basename(file)}")

    def choose_sets(self):
        self.display_files()
        i = int(input(f"Training file #:")) - 1
        self.display_files()
        j = int(input(f"Testing file #:")) - 1
        
        return self.files[i], self.files[j]

dfg = DataframeGenerator("pre-data")
print(dfg.train.head())
print(dfg.test.head())
