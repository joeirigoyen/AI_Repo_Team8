import sys
import os
import pandas as pd
from data_handler import DataHandler
from random_forest import RFClassifier


if __name__ == '__main__':
    train_path = os.path.join(sys.path[0], 'train.csv')
    sample_path = os.path.join(sys.path[0], 'test.csv')
    output_path = os.path.join(sys.path[0], 'submission.csv')

    handler = DataHandler(train_path)
    sample_dataframe = pd.read_csv(sample_path)

    train_df = handler.df
    train_x_columns = train_df.drop('Transported', axis=1).columns

    train_x = train_df.drop('Transported', axis=1).to_numpy()
    train_y = train_df.drop(train_x_columns, axis=1).to_numpy()

    sample = handler.process_data(sample_dataframe, 0, sample=True).to_numpy()

    rfc = RFClassifier(train_x, train_y.ravel(), 100, "entropy", 15, 2, 1, None)
    rfc.adjust()
    results = rfc.predict(sample)

    passengers = sample_dataframe['PassengerId']
    predicts = pd.Series(results).apply(lambda x : True if x == 1 else False)

    results_df = pd.DataFrame(passengers)
    results_df.insert(results_df.shape[1], 'Transported', predicts)

    results_df.to_csv(output_path, index=False)
