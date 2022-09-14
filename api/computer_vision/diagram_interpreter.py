import base64
import json
from pathlib import Path
from sre_parse import CATEGORIES
import cv2 as cv
import sys
import numpy as np
from flask import Response, jsonify
from computer_vision.diagram_interpreter_meta import DiagramInterpreterMeta
from computer_vision.label import Label
from computer_vision.shape import Shape
from collections import defaultdict
from pytesseract import *
from shapely.geometry import Point, Polygon
from keras.models import load_model
from keras import Sequential
from tensorflow import keras

font = cv.FONT_HERSHEY_COMPLEX
specified_width = 100
specified_height = 100
CNN_MODEL_LOCATION = 'computer_vision/cnn_model.h5'
CATEGORIES = ['Circle', 'Rectangle', 'star', 'Triangle']

class DiagramInterpreter(metaclass=DiagramInterpreterMeta):

    cnn_model: Sequential = None
    #shape_repository: list = None

    # Tesseract config - Fully automatic page segmentation (Default option)
    # Source: https://newbedev.com/pytesseract-ocr-multiple-config-options
    custom_config = r'--psm 3'

    def __init__(self) -> None:
        self.cnn_model = load_model(CNN_MODEL_LOCATION)

    def __prepare_image(self, img):
        # convert to grayscale
        img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(img_grayscale, (3, 3), cv.BORDER_DEFAULT)
        # Convert the grayscale image to binary (image binarization opencv python)
        _, binary_img = cv.threshold(blurred, 250, 255, cv.THRESH_BINARY)
        # Invert image
        inverted_binary_img = ~ binary_img

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (6, 6), (-1, -1))
        nodes = cv.morphologyEx(inverted_binary_img, cv.MORPH_OPEN, kernel)

        vertexes, _ = cv.findContours(nodes, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for vertex_num in range(len(vertexes)):
            selected_contour = vertexes[vertex_num]
            
            if (cv.contourArea(selected_contour) > 1000):
                cv.fillPoly(nodes, pts =[selected_contour], color=(255,255,255))
            else:
                cv.fillPoly(nodes, pts =[selected_contour], color=(0,0,0))
            
        nodes_larger = cv.morphologyEx(nodes, cv.MORPH_DILATE, kernel)
        links = cv.subtract(inverted_binary_img, nodes_larger)

        nodes_larger = cv.morphologyEx(nodes_larger, cv.MORPH_DILATE, kernel)
        intersections = cv.bitwise_and(nodes_larger, links)

        return (nodes, nodes_larger, links, intersections)

    def __detect_shapes(self, vertexes, nodes):
        shapes = []
        inverted_node_image = ~nodes
        for vertex_num in range(len(vertexes)):
            selected_contour = vertexes[vertex_num]
            end_points = (cv.approxPolyDP(selected_contour, 0.01 *
                          cv.arcLength(selected_contour, True), True)).tolist()
            cleaned_end_points = []
            for ep in end_points:
                cleaned_end_points.append(ep[0])
            M = cv.moments(selected_contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            x, y, width, height = cv.boundingRect(selected_contour)

            if (cv.contourArea(selected_contour) > 1000):
                lowest_rating = 1
                ROI = inverted_node_image[y-20:y+height+20, x-20:x+width+20]
            
                rescaled_image = cv.resize(ROI,(specified_width, specified_height), interpolation=cv.INTER_LINEAR)

                image_array = keras.utils.img_to_array(rescaled_image)
                image_array = np.round(image_array/255,3).copy()

                image_array = np.expand_dims(image_array, axis = 0)

                prediction = self.cnn_model.predict(image_array)
                list_of_predictions = list(np.round(prediction[0]))
                predicted_shape = CATEGORIES[list_of_predictions.index(1)]
                new_shape = Shape(vertex_num, predicted_shape, cleaned_end_points, (cX,cY), width, height)
                shapes.append(new_shape)
        return shapes
    def __detect_connections(self, intersection_list: list, edge_list: list, enlarged_vertex_list: list):
        vertex_edge_list = []
        connected_vertex_list = []
        grouped_edges = defaultdict(list)
        # for every intersection check if any point inside an intersection connects to a certain vertex and edge
        # after finding a vertex - edge pair, it is put inside a list
        for intersection in intersection_list:
            edge_index = ''
            vertex_index = ''
            for n in range(len(enlarged_vertex_list)):
                if(any(cv.pointPolygonTest(enlarged_vertex_list[n], (int(point[0][0]),int(point[0][1])) , False)  >= 0 for point in intersection)):
                    vertex_index = n

            for e in range(len(edge_list)):
                if(any(cv.pointPolygonTest(edge_list[e], (int(point[0][0]),int(point[0][1])) , False)  >= 0 for point in intersection)):
                    edge_index= e
            
            vertex_edge_list.append((vertex_index, edge_index))

        # the list is grouped by edges
        for ve in vertex_edge_list:
            grouped_edges[ve[1]].append(ve)

        # the vertexes connected to a edge are checked and if there are only two vertexes, the connection is valid and added to the list of connections
        for edge in grouped_edges:
            connection = []
            for vertex in grouped_edges[edge]:
                connection.append(vertex[0])
            if len(connection) == 2:
                connected_vertex_list.append(connection)

        return connected_vertex_list
    
    # Scans grayscale image for text using Tesseract OCR. 
    # The text is input in a dictionary and if the vale stripped of whitespaces is longer than 0,\
    # a Label object is created and added to the label list.
    def __detect_text(self, img):
        img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        results = pytesseract.image_to_data(img_grayscale,config=self.custom_config,  output_type=Output.DICT)
        detected_text = []
        for i in range(0, len(results["text"])):
            if(len(results["text"][i].strip())):
                detected_text.append(Label(results['text'][i], results['left'][i],results['top'][i]))
        return detected_text
    

    def get_diagram_elements(self, image_string):
        
            outer_labels = []
            inner_labels = []
            array = base64.b64decode(image_string)
            np_array = np.frombuffer(array, dtype= np.uint8)
            img = cv.imdecode(np_array, cv.IMREAD_UNCHANGED)
            if img is None:
                sys.exit("Could not read the image")
            img_copy = img.copy()

            detected_labels = self.__detect_text(img_copy)
            processed_images = self.__prepare_image(img_copy)

            vertexes, _ = cv.findContours(processed_images[0], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            enlarged_vertex_list, _ = cv.findContours(processed_images[1], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            edge_list, _ = cv.findContours(processed_images[2], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            intersection_list, _ = cv.findContours(processed_images[3], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            shapes = self.__detect_shapes(vertexes, processed_images[0])
            connections = self.__detect_connections(intersection_list, edge_list,enlarged_vertex_list)
            
            for shape in shapes:
                label_list = []
                for connection in connections:
                    if shape.id in connection:
                        if(connection[0] == shape.id):
                            shape.connections.append(connection[1])
                        else:
                            shape.connections.append(connection[0])

                for label in detected_labels:
                    point = Point(label.x, label.y)
                    polygon = Polygon(shape.end_points)
                    if(polygon.contains(point)):
                        label_list.append(label.value)
                        inner_labels.append(label)
                shape.text = ' '.join(label_list)

            outer_labels = [x for x in detected_labels if not x in inner_labels]

            shapes_dictionary = list(map(lambda shape: shape.asdict(), shapes))    
            label_dictionary = list(map(lambda label: label.asdict(), outer_labels))

            data = {'ShapeList': shapes_dictionary, 'OuterLabels': label_dictionary}
            return json.dumps(data)


