import os
import numpy as np
import pandas as pd
import df_generator as dfg

data_file = os.path.join(os.path.abspath(os.path.curdir), 'data', 'train.csv')

df_gen = dfg.DataframeGenerator(data_file)
df = df_gen.df

def compute_smd(data_1, data_2):
    data_1 = np.array(data_1)
    data_2 = np.array(data_2)
    return np.abs((data_1.mean() - data_2.mean())) / np.sqrt((np.power(data_1.std(), 2) + np.power(data_2.std(), 2)) / 2)


print(compute_smd(pd.DataFrame([1, 2, 3, 4, 5]), pd.DataFrame([2, 3, 4, 5, 6])))
