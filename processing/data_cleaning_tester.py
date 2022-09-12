import os
import numpy as np
import pandas as pd
import seaborn as sb
import df_generator as dfg
import matplotlib.pyplot as plt

data_file = os.path.join(os.path.abspath(os.path.curdir), 'data', 'train.csv')
output_file = os.path.join(os.path.abspath(os.path.curdir), 'data', 'current_df.csv')

df_gen = dfg.DataframeGenerator(data_file)
df = df_gen.df


def compute_smd(data_1, data_2):
    data_1 = np.array(data_1)
    data_2 = np.array(data_2)
    return np.abs((data_1.mean() - data_2.mean())) / np.sqrt((np.power(data_1.std(), 2) + np.power(data_2.std(), 2)) / 2)

""" sb.heatmap(df.corr(), annot=True)
plt.show() """
df.to_csv(output_file, index=False)
