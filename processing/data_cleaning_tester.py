import sys
from data_cleaner import DataframeGenerator


# Get original data
dfg = DataframeGenerator("pre-data")

# Get training and testing data
train = dfg.train
test = dfg.test
