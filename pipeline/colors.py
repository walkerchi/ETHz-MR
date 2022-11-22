import numpy as np
class Colors:
    ORANGE = np.array([255,128,0],dtype=np.float32) / 255
    RED    = np.array([255, 0, 0],dtype=np.float32) / 255
    GREEN  = np.array([0, 255, 0],dtype=np.float32) / 255
    BLUE   = np.array([0,   0,255],dtype=np.float32) / 255
    BLACK  = np.array([0, 0, 0],dtype=np.float32)
    WHITE  = np.array([255,255,255],dtype=np.float32) / 255
    DARK_GRAY = np.array([50,50,50],dtype=np.float32) / 255

COLORS_DICT = {
    "orange":Colors.ORANGE,
    "red":Colors.RED,
    "green":Colors.GREEN,
    "blue":Colors.BLUE,
    "black":Colors.BLACK
}