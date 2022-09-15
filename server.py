import os
import csv
from flask import Flask, request, jsonify

from processing.data_handler import DataHandler
from processing.svm import SupportVectorMachine
from processing.random_forest import RFClassifier
from processing.logistic_regression import LogisticRegressionModel
from joblib import load

server = Flask(__name__)

@server.route('/test', methods=['GET'])
def test():
    return jsonify({"Message": "Welcome"}), 200


@server.route('/form', methods=['POST'])
def input_data():
    passengers = request.json["Passengers"]
    model = request.json["Model"]
    df_header = ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name']
    dir_path = os.path.join(os.path.abspath(os.path.curdir), "tmp")
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    path = os.path.join(dir_path, "test.csv")
    with open(path, 'w', encoding='UTF8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=df_header)
        writer.writeheader()
        for passenger in passengers:
            writer.writerow(passenger)

    test_path= os.path.join(os.path.abspath(os.path.curdir), "tmp", "test.csv")
    train_path= os.path.join(os.path.abspath(os.path.curdir), "data", "train.csv")
    dh = DataHandler(train_path)
    print("here1")
    sample = dh.process_sample(test_path)
    print("here2")
    """ lr = load('logistic_regression.joblib') 
    rf = load('random_forest.joblib') 
    svm = load('svm.joblib')
    print("here3")
    if (model == 'LogisticRegression'):
        res = lr.predict(sample)
    elif (model == 'RandomForest'):
        res = rf.predict(sample)
    elif (model == 'SVM'):
        res = svm.predict(sample)

    results = []
    for elem in res:
        results.append(int(elem))
    print("here4")
    for i in range(len(passengers)):
        passengers[i]["Result"] = results[i]
    print("here5") """
    return jsonify({"Message": "Success", "Passengers": passengers}), 200


@server.route('/init', methods=['POST'])
def initialize_data():
    train_path= os.path.join(os.path.abspath(os.path.curdir), "data", "train.csv")
    dh = DataHandler(train_path)
    df = dh.df
    x = df.drop('Transported', axis=1)
    print(x.shape)
    y = df['Transported']
    svm = SupportVectorMachine(x, y, 1000)
    svm.fit()
    rf = RFClassifier(x, y, 25, 'log_loss', 8, 2, 2, 'log2')
    rf.adjust()
    lr = LogisticRegressionModel(x, y, 42, 1000)
    lr.adjust()

    return jsonify({"message": "Success"}), 200


if __name__ == '__main__':
    server.run(debug=True,host='0.0.0.0',port='8080')
