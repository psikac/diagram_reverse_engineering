import json
from flask import jsonify, request
from flask_restful import Resource

from computer_vision.diagram_interpreter import DiagramInterpreter

class ImageController(Resource):

    def post(self):
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
          data = request.json['image']
          interpreter = DiagramInterpreter()
          results = interpreter.get_diagram_elements(data)
          if(results != None):
            return results
        else:
            return 'Content-Type not supported!'