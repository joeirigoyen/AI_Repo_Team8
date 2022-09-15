import os
import numpy as np
import pandas as pd
import seaborn as sb
import data_handler as dh
import matplotlib.pyplot as plt

data_file = os.path.join(os.path.abspath(os.path.curdir), 'data', 'train.csv')
output_file = os.path.join(os.path.abspath(os.path.curdir), 'data', 'current_df.csv')
data_location = os.path.join(os.path.abspath(os.path.curdir), 'data')

handler = dh.DataHandler(data_file)
df = handler.df


def compute_smd(data_1, data_2):
    data_1 = np.array(data_1)
    data_2 = np.array(data_2)
    return np.abs((data_1.mean() - data_2.mean())) / np.sqrt((np.power(data_1.std(), 2) + np.power(data_2.std(), 2)) / 2)

sb.heatmap(df.corr(), annot=True)
# Get data before resampling
data_x = df.drop('Transported', axis=1)
data_y = df['Transported']
# Plot data
plt.show()
df.to_csv(output_file, index=False)


# Get test sample
""" x_test = handler.process_data(os.path.join(data_location, 'test.csv'), 0, sample=True)
print(x_test.isna().value_counts())
x_test.to_csv(os.path.join(data_location, 'test_processed.csv'), index=False) """

""" # Test group count
data_x = df.drop('Transported', axis=1)
data_y = df['Transported']
groups = pd.Series(data_x['Group'].unique()).values
members = pd.Series(data_x.groupby('Group')['Group'].value_counts().values).values

group_dict = {}

for group, num in zip(groups, members):
    group_dict[group] = num

data_x['GroupMembers'] = data_x['Group'].apply(lambda g : group_dict[g]) """