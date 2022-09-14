class Label:
    def __init__(self, value, x, y):
        self.value = value
        self.x = x
        self.y = y

    def dump(self):
        return {"LabelList": {'value': self.value,
                               'x': self.x,
                               'y': self.y}}
    def asdict(self):
        return {'value': self.value, 'x':self.x, 'y':self.y}