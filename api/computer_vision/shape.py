class Shape:
    def __init__(self, id, shape, end_points, center_coordinates, width, height):
        self.id = id
        self.shape = shape
        self.end_points = end_points
        self.center_coordinates = center_coordinates
        self.width = width
        self.height = height
        self.connections = []
        self.text = ""

    def asdict(self):
        return { 
            'id': self.id, 
            'shape': self.shape,  
            'connections': self.connections, 
            'text':self.text, 
            'center_coordinates': self.center_coordinates,
            'width': self.width,
            'height': self.height}

    def tostring(self):
        return { 
            'id': self.id, 
            'shape': self.shape,  
            'connections': self.connections, 
            'text':self.text, 
            'center_coordinates': self.center_coordinates,
            'width': self.width,
            'height': self.height}
