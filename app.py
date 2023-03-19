from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from main import elevation_to_colour
import requests
# import torch

DEBUG = True
app = Flask(__name__)
CORS(app)

@app.route('/ping', methods=['GET'])
def pong():
    return jsonify('pong!')


@app.route('/predict', methods=['POST', 'GET'])
def make_prediction():
    event_data = request.json
    print(type(event_data))

    input = np.array(event_data)
    print(input.shape)
    print(input)
    elevation_to_colour(input, True)
    # print(event_data)
    # print("")
    return jsonify('wowo')
