from collections import defaultdict
from doctest import OutputChecker
import cv2 as cv
import numpy as np
from pathlib import Path
import sys
from itertools import islice
from pytesseract import*
import json
import csv
from shapely.geometry import Point, Polygon
from computer_vision.label import Label
from computer_vision.shape import Shape
from computer_vision.shape_handler import RectangleHandler, ShapeHandler, TriangleHandler
from keras.models import load_model
from tensorflow import keras

""" triangle_handler: ShapeHandler = TriangleHandler()
rectangle_handler: ShapeHandler = RectangleHandler()
triangle_handler.set_next(rectangle_handler) """

specified_width = 100
specified_height = 100
cnn_model = load_model('computer_vision/cnn_model.h5')
print(type(cnn_model))
categories = ['circle', 'rectangle', 'star', 'triangle']

def prepare_image(img):
        #convert to grayscale
        img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        cv.imwrite("images/img_grayscale.png", img_grayscale)

        blurred = cv.GaussianBlur(img_grayscale,(3,3),cv.BORDER_DEFAULT)
        #get threshold value of objects
        #threshold_value = img_grayscale[350, 270]
        # Convert the grayscale image to binary (image binarization opencv python)
        _, binary_img = cv.threshold(blurred, 250, 255, cv.THRESH_BINARY)

        cv.imwrite("images/img_binary.png", binary_img)

        # Invert image
        inverted_binary_img = ~ binary_img

        cv.imwrite("images/img_inverted_binary.png", inverted_binary_img)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (6,6), (-1, -1))
        objects = cv.morphologyEx(inverted_binary_img, cv.MORPH_OPEN, kernel)
        cv.imwrite("images/objects.png", objects)


        vertexes, _ = cv.findContours(objects, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for vertex_num in range(len(vertexes)):
            selected_contour = vertexes[vertex_num]
            
            if (cv.contourArea(selected_contour) > 1000):
                cv.fillPoly(objects, pts =[selected_contour], color=(255,255,255))
            else:
                cv.fillPoly(objects, pts =[selected_contour], color=(0,0,0))

        # kernel = np.ones((5,5),np.uint8) 

        cv.imwrite("images/objects_filled.png", objects)

     

        #kernel = np.ones((5,5),np.uint8)
        nodes_larger = cv.morphologyEx(objects, cv.MORPH_DILATE, kernel)
        cv.imwrite("images/nodes_larger.png", nodes_larger)
        links = cv.subtract(inverted_binary_img, nodes_larger)
        cv.imwrite("images/links.png", links)
        nodes_larger = cv.morphologyEx(nodes_larger, cv.MORPH_DILATE, kernel)
        cv.imwrite("images/nodes_larger2.png", nodes_larger)
        intersections = cv.bitwise_and(nodes_larger, links)
        cv.imwrite("images/intersections.png", intersections)

        return (objects, nodes_larger, links, intersections)

        
def detect_shapes(vertexes):
    shapes = []
    black_on_white = ~processed_images[0]
    #vertexes = list(filter(lambda v: cv.contourArea(v) > 1000, vertexes))
    print(len(vertexes))
    for vertex_num in range(len(vertexes)):
        selected_contour = vertexes[vertex_num]
        #print(end_points)
        if(cv.contourArea(selected_contour) > 1000):
            end_points = (cv.approxPolyDP(selected_contour, 0.01 * cv.arcLength(selected_contour, True), True)).tolist()
            cleaned_end_points = []
            for ep in end_points:
                cleaned_end_points.append(ep[0])
            M = cv.moments(selected_contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            x,y,width,height = cv.boundingRect(selected_contour)
            print(vertex_num)
            #print(f"Index num: {vertex_num}")
            lowest_rating = 1
            selected_shape = []
            cv.fillPoly(black_on_white, pts =[selected_contour], color=(0,0,0))
            ROI = black_on_white[y-20:y+height+20, x-20:x+width+20]
            cv.imwrite(f"images/{vertex_num}.png", ROI)

            #print(len(end_points))
            """ for shape in shape_repository:
                #print(shape[0])
                ret = cv.matchShapes(selected_contour,shape[1],1,0.0)
                #print(ret)
                if(ret<lowest_rating):
                    lowest_rating = ret
                    selected_shape = shape """

            rescaled_image = cv.resize(ROI,(specified_width, specified_height), interpolation=cv.INTER_LINEAR)

            cv.imshow('First detected contour', ROI)
            cv.waitKey(0)
            cv.destroyAllWindows()

            image_array = keras.utils.img_to_array(rescaled_image)
            image_array = np.round(image_array/255,3).copy()

            image_array = np.expand_dims(image_array, axis = 0)

            prediction = cnn_model.predict(image_array)

            list_of_predictions = list(np.round(prediction[0]))
            """ print(list_of_predictions)
            print(categories[list_of_predictions.index(1)]) """
            predicted_shape = categories[list_of_predictions.index(1)]
            new_shape = Shape(vertex_num, predicted_shape, cleaned_end_points, (cX,cY), width, height)
            shapes.append(new_shape)
            #print(new_shape.tostring())
            """  shape = triangle_handler.handle_shape(vertex_num, cleaned_end_points)
            if (shape != None):
                shapes.append(shape) """
    return shapes

def detect_connections(intersection_list: list, edge_list: list, enlarged_vertex_list: list):
    vertex_edge_list = []
    connected_vertex_list = []
    grouped_edges = defaultdict(list)
    print(len(enlarged_vertex_list))
    print()
    #for every intersection check if any point inside an intersection connects to a certain vertex and edge
    #after finding a vertex - edge pair, it is put inside a list
    for intersection in intersection_list:
        edge_index = ''
        vertex_index = ''
        for n in range(len(enlarged_vertex_list)):
            if(any(cv.pointPolygonTest(enlarged_vertex_list[n], (int(point[0][0]),int(point[0][1])) , False)  >= 0 for point in intersection) and cv.contourArea(enlarged_vertex_list[n]) > 1000):
                vertex_index = n

        for e in range(len(edge_list)):
            if(any(cv.pointPolygonTest(edge_list[e], (int(point[0][0]),int(point[0][1])) , False)  >= 0 for point in intersection)):
                edge_index= e
        
        vertex_edge_list.append((vertex_index, edge_index))
        print((vertex_index, edge_index))

    #the list is grouped by edges
    for ve in vertex_edge_list:
        grouped_edges[ve[1]].append(ve)

    #the vertexes connected to a edge are checked and if there are only two vertexes, the connection is valid and added to the list of connections
    for edge in grouped_edges:
        connection = []
        for vertex in grouped_edges[edge]:
            connection.append(vertex[0])
        if len(connection) == 2:
            connected_vertex_list.append(connection)

    return connected_vertex_list

#Scans grayscale image for text using Tesseract OCR. 
#The text is input in a dictionary and if the vale stripped of whitespaces is longer than 0,\
# a Label object is created and added to the label list.
def detect_text(img):
    img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    results = pytesseract.image_to_data(img_grayscale,config=custom_config,  output_type=Output.DICT)
    detected_text = []
    for i in range(0, len(results["text"])):
        if(len(results["text"][i].strip())):
            detected_text.append(Label(results['text'][i], results['left'][i],results['top'][i]))

    for text in detected_text:
        print(text.asdict())
    return detected_text


def load_shape_repository(directory):
    shape_repository = []
    for file in Path(directory).iterdir():
        if(file.is_file()):
            # print(file.absolute())
            img = cv.imread(file.as_posix())
            img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            blurred = cv.GaussianBlur(img_grayscale,(3,3),cv.BORDER_DEFAULT)
            _, binary_img = cv.threshold(blurred, 120, 255, cv.THRESH_BINARY)
            inverted_binary_img = ~ binary_img
            """ cv.imshow('First detected contour', inverted_binary_img)
            cv.waitKey(0)
            cv.destroyAllWindows() """
            vertexes, _ = cv.findContours(inverted_binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            shape_repository.append(
                [
                    file.name.split('.',1)[0],  
                    vertexes[0], 
                    len(cv.approxPolyDP(vertexes[0], 0.01 * cv.arcLength(vertexes[0], True), True))
                ])
    return shape_repository


file_separator = '.'
directory = "shape_repository"
font = cv.FONT_HERSHEY_COMPLEX

pathString = 'resources/test_image_3.png'
path = Path(pathString)
#Tesseract config - Fully automatic page segmentation (Default option)
#Source: https://newbedev.com/pytesseract-ocr-multiple-config-options
custom_config = r'--psm 3'


#shape_repo = load_shape_repository(directory)
# print(len(shape_repo))
""" for shape in shape_repo:
    print(shape[0]) """
img = cv.imread(pathString)

if img is None:
    sys.exit("Could not read the image")

img_copy = img.copy()


processed_images = prepare_image(img_copy)


detected_labels = detect_text(img_copy)
vertexes, _ = cv.findContours(processed_images[0], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
enlarged_vertex_list, _ = cv.findContours(processed_images[1], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
edge_list, _ = cv.findContours(processed_images[2], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
intersection_list, _ = cv.findContours(processed_images[3], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

shapes = detect_shapes(vertexes)
connections = detect_connections(intersection_list, edge_list,enlarged_vertex_list)
outer_labels = []
inner_labels = []

for shape in shapes:
    label_list = []
    for connection in connections:
        #print(shape.id)
        if shape.id in connection:
            print(connection)
            #shape.connections.append(list(filter(lambda con: con != shape.id, connection))[0])
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
json3 = json.dumps(data)
print(json3)
""" except Exception as e:
    if hasattr(e, 'message'):
        print(e.message)
    else:
        print(e) """