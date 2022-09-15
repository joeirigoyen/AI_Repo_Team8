# Models to be used as weak learners
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# Boosters
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
# Model selectors
from sklearn.model_selection import StratifiedKFold, cross_val_score
from skopt import BayesSearchCV
# Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
# Pipeline 
from sklearn.pipeline import Pipeline
# Dataframe and samples handler
from data_handler import DataHandler as dh
# Miscelaneous imports
import os
import pandas as pd


# Declare pipeline list
pipelines = [Pipeline([('GaussianNB', GaussianNB())]),
            Pipeline([('KNeighborsClassifier', KNeighborsClassifier())]),
            Pipeline([('RandomForestClassifier', RandomForestClassifier())]),
            Pipeline([('DecisionTreeClassifier', DecisionTreeClassifier())]),
            Pipeline([('XGBClassifier', XGBClassifier())]),
            Pipeline([('LGBMClassifier', LGBMClassifier())]),
            Pipeline([('LogisticRegression', LogisticRegression(solver='liblinear', max_iter=3000))]),
            Pipeline([('AdaBoostClassifier', AdaBoostClassifier())]),
            Pipeline([('CatBoostClassifier', CatBoostClassifier(verbose=0, one_hot_max_size=10))])]

# Declare parent data directory
data_location = os.path.join(os.path.abspath(os.path.curdir), 'data')

# Get processed data
handler = dh(os.path.join(data_location, 'train.csv'))
train_df = handler.df
x_train = train_df.drop('Transported', axis=1)
y_train = train_df['Transported']

# Fit for each pipeline in list
for pipe in pipelines:
    pipe.fit(x_train, y_train)

# Get accuracy from each classifier
pipe_results = {}
classifier_finder = {
    0 : 'GaussianNB',
    1 : 'KNeighborsClassifier',
    2 : 'RandomForestClassifier',
    3 : 'DecisionTreeClassifier',
    4 : 'XGBClassifier',
    5 : 'LGBMClassifier',
    6 : 'LogisticRegression',
    7 : 'AdaBoostClassifier',
    8 : 'CatBoostClassifier'
}

for index, model in enumerate(pipelines):
    score = cross_val_score(model, x_train.values, y_train.values.ravel(), cv=5, scoring='accuracy').mean()
    pipe_results[classifier_finder[index]] = score

for key, value in sorted(pipe_results.items(), key=lambda item : item[1]):
    print(key, value)

# Apply grid search on the top performing models
""" cat_booster = CatBoostClassifier(eval_metric='accuracy', verbose=0)
cat_booster_params = {
    'learning_rate' : [0.001, 0.005, 0.01, 0.02, 0.03],
    'depth' : [5, 6, 7, 8, 10],
    'iterations' : [500, 750, 1000, 1250, 2000]
} """

""" cv = StratifiedKFold(n_splits=5, shuffle=True)
bayes_search = BayesSearchCV(estimator=cat_booster, search_spaces=cat_booster_params, scoring='Accuracy', cv=cv, n_jobs=-1, n_iter=3000, verbose=0, refit=True).fit(x_train, y_train)
best_cat_params = bayes_search.get_params()
print(best_cat_params) """

# Get test sample
test_passengers = pd.read_csv(os.path.join(data_location, 'test.csv'))['PassengerId']
x_test = handler.process_sample(os.path.join(data_location, 'test.csv'))
x_test.to_csv(os.path.join(data_location, 'test_processed.csv'), index=False)

# Make predictions with best model
best_model = pipelines[-1]
predictions = best_model.predict(x_test.values)

# Create submission file
submission = pd.DataFrame()
submission['PassengerId'] = test_passengers
submission['Transported'] = predictions
submission['Transported'] = submission['Transported'].apply(lambda val : True if val == 1 else False)
submission.to_csv(os.path.join(data_location, 'submission.csv'), index=False)
