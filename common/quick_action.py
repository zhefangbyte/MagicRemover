from enum import Enum


class QuickActionMode(Enum):
    UNKNOWN = 0
    BRUSH = 1
    REMOVE_BACKGROUND = 2
    UNDO = 3
    REDO = 4
    ROTATE = 5
    ZOOM_IN = 6
    ZOOM_OUT = 7
    SCALE_TO_WINDOW = 8
    SCALE_TO_RAW = 9
    COLOR_OVERLAY = 10
    IMAGE_OVERLAY = 11
    PURIFY_BACKGROUND = 12
    TEXT = 13

class QuickAction:
    def __init__(self, mode: QuickActionMode, takeTime: bool = False):
        self.mode = mode
        self.takeTime = takeTime