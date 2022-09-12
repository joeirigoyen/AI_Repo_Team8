import os
from flask import Flask, request, jsonify
import csv

from processing.df_generator import DataHandler

server = Flask(__name__)


@server.route('/form', methods=['POST'])
def input_data():
    passengers = request.json["Passengers"]
    df_header = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'Shopping', 'Spa', 'VRDeck', 'Name']
    path = os.path.join("tmp", "test.csv")
    with open(path, 'w', encoding='UTF8', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(df_header)
                for i in range(len(passengers)):
                    input_data = passengers[i].values()
                    writer.writerow(input_data)
    dh = DataHandler(path)
    return jsonify({"message": "Success"}), 200

if __name__ == '__main__':
    server.run(debug=True,host='0.0.0.0',port='8080')