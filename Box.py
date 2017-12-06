class Box:
    
    x = 0
    y = 0
    w = 0
    h = 0
    cast = 0
    prob = 0.0
    
    def __init__(self, x, y, w, h, cast, prob):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.cast = cast
        self.prob = prob
