import os
import numpy as np
import pandas as pd
import seaborn as sb
import data_handler as dh
import matplotlib.pyplot as plt

data_file = os.path.join(os.path.abspath(os.path.curdir), 'data', 'train.csv')
output_file = os.path.join(os.path.abspath(os.path.curdir), 'data', 'current_df.csv')

df_gen = dh.DataHandler(data_file)
df = df_gen.df


def compute_smd(data_1, data_2):
    data_1 = np.array(data_1)
    data_2 = np.array(data_2)
    return np.abs((data_1.mean() - data_2.mean())) / np.sqrt((np.power(data_1.std(), 2) + np.power(data_2.std(), 2)) / 2)

sb.heatmap(df.corr(), annot=True)
# Get data before resampling
data_x = df.drop('Transported', axis=1)
data_y = df['Transported']
# Get data after resampling
rs_data = df_gen.oversample_data(df, 'Transported')
rs_data_y = rs_data['Transported']
# Plot data
sb.kdeplot(data_y, label='Original', color='red')
sb.kdeplot(rs_data_y, label='Oversampled', color='blue')
plt.show()
df.to_csv(output_file, index=False)
