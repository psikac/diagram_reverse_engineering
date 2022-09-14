from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_restful import Resource, Api
from controllers.image_controller import ImageController

app = Flask(__name__)
api = Api(app)
CORS(app)

api.add_resource(ImageController, '/image')

if __name__ == '__main__':
    app.run(debug=True)
    