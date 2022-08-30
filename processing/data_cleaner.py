import os
import pandas as pd


class DataframeGenerator:
    def __init__(self, directory: str, encoding="utf-8", na_values='?') -> None:
        # Set training and testing files --- NOTE: The lines below will only be used for developing purposes and it's not the actual implementation
        self.test = self.clean_data(pd.read_csv("pre-data\\test.csv"))
        self.train = self.clean_data(pd.read_csv("pre-data\\train.csv"))
        # --- DO NOT DELETE: These are the actual lines of code that will be ran in the final version ---
        """ # Initialize file list
        self.files = []
        # Get list of files in directory
        for file in os.listdir(directory):
            self.files.append(os.path.abspath(os.path.join(directory, file)))
        # Let user decide which set to use for training and testing
        self.train, self.test = self.choose_sets()
        self.train = self.clean_data(pd.read_csv(self.train, encoding=encoding, na_values=na_values))
        self.test = self.clean_data(pd.read_csv(self.test, encoding=encoding, na_values=na_values)) """

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # For now, just remove the na values from the data
        return df.dropna()

    def display_files(self) -> None:
        print(f"Files in data directory:")
        for index, file in enumerate(self.files):
            print(f"{index + 1}) {os.path.basename(file)}")

    def choose_sets(self) -> tuple:
        self.display_files()
        i = int(input(f"Training file #:")) - 1
        self.display_files()
        j = int(input(f"Testing file #:")) - 1

        return self.files[i], self.files[j]