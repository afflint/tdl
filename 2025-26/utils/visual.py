import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

def process_image(image_path):
    image = Image.open(image_path).resize((64, 64))
    image_matrix = np.array(image)
    
    red_channel = image_matrix[:, :, 0]  # Canale Rosso
    green_channel = image_matrix[:, :, 1]  # Canale Verde
    blue_channel = image_matrix[:, :, 2]  # Canale Blu

    return image_matrix, red_channel, green_channel, blue_channel