import numpy as np

class ObjectCoords:
    
    def __init__(self, xmin, ymin, xmax, ymax) -> None:
        super().__init__()
        self.xmin = xmin 
        self.ymin = ymin 
        self.xmax = xmax 
        self.ymax = ymax
        self.w = xmax - xmin
        self.h = ymax - ymin
        self.xc = (xmin + xmax) / 2
        self.yc = (ymin + ymax) / 2

def gaussian_radius(output_width, output_height, xcenter, ycenter, width, height):

    xrange = np.arange(output_width) - xcenter
    yrange = np.arange(output_height) - ycenter
    
    xrange_scaled = xrange / (width / 10)
    yrange_scaled = yrange / (height / 10)
    
    heatmap_xrange = np.exp(-((xrange_scaled ** 2) / 2)).reshape(1, -1)
    heatmap_yrange = np.exp(-((yrange_scaled ** 2) / 2)).reshape(-1, 1)
    
    heatmap = heatmap_xrange * heatmap_yrange

    return heatmap

