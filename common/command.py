from enum import Enum


class CommandMode(Enum):
    UNKNOWN = 0
    RECTANGLE_SELECT_REMOVE = 1
    PAINT_BUCKET = 3
    DRAW = 4
    CROP = 5
    DROPPER = 6
    POINT_SELECT_REMOVE = 7
    PAINT_SELECT_REMOVE = 8

class Command:
    def __init__(self, mode: CommandMode, takeTime: bool = False):
        self.mode = mode
        self.takeTime = takeTime
