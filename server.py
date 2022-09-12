import os
import csv
from flask import Flask, request, jsonify


from processing.df_generator import DataHandler

server = Flask(__name__)


@server.route('/form', methods=['POST'])
def input_data():
    passengers = request.json["Passengers"]
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

    train_path= os.path.join(os.path.abspath(os.path.curdir), "data", "train.csv")
    test_path= os.path.join(os.path.abspath(os.path.curdir), "tmp", "test.csv")
    dh = DataHandler(train_path)
    sample = dh.process_data(test_path, 0, sample=True)
    print(sample.head())
    return jsonify({"message": "Success"}), 200

if __name__ == '__main__':
    server.run(debug=True,host='0.0.0.0',port='8080')
