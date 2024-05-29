from enum import Enum

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QFile, Qt
from PyQt5.QtGui import QPixmap, QCursor, QImage
from qfluentwidgets import getIconColor, Theme, FluentIconBase


class FluentIconExt(FluentIconBase, Enum):
    SELECT_OBJECT = 'select_object'
    ARROW_UNDO = 'arrow_undo'
    ARROW_REDO = 'arrow_redo'
    FLUENT_VIDEO = 'fluent_video'
    VIDEO_BACKGROUND_EFFECT = 'video_background_effect'
    RESIZE_LARGE = 'resize_large'
    PAINT_BUCKET = 'paint_bucket'
    CROP = 'crop'
    CALENDAR_CANCEL = 'calendar_cancel'
    ARROW_EXPORT_LTR = 'arrow_export_ltr'
    IMAGE_SHADOW = 'image_shadow'
    COLOR_BACKGROUND = 'color_background'
    INKING_TOOL = 'inking_tool'
    IMAGE_ARROW_COUNTERCLOCKWISE = 'image_arrow_counterclockwise'
    EYEDROPPER = 'eyedropper'
    HAND_DRAW = 'hand_draw'
    DUAL_SCREEN_MIRROR = 'dual_screen_mirror'
    GRID_DOTS = 'grid_dots'

    def path(self, theme=Theme.AUTO):
        return f'resource/icon/ic_fluent_{self.value}_24_regular_{getIconColor(theme)}.svg'


class FluentIconFilledExt(FluentIconBase, Enum):
    PAINT_BUCKET = 'paint_bucket'
    INKING_TOOL = 'inking_tool'

    def path(self, theme=Theme.AUTO):
        return f'resource/icon/ic_fluent_{self.value}_24_filled_{getIconColor(theme, reverse=True)}.svg'


class CursorIcon(FluentIconBase, Enum):
    ALTERNATE = 'alternate'
    BEAM = 'beam'
    DGN1 = 'dgn1'
    DGN2 = 'dgn2'
    HANDWRITING = 'handwriting'
    HELP = 'help'
    HORZ = 'horz'
    LINK = 'link'
    MOVE = 'move'
    PERSON = 'person'
    PIN = 'pin'
    POINTER = 'pointer'
    PRECISION = 'precision'
    UNAVAILABLE = 'unavailable'
    VERT = 'vert'

    def path(self, theme=Theme.AUTO):
        return f'resource/icon/cursor/{self.value}_{getIconColor(theme, reverse=True)}.cur'

    def create(self, theme=Theme.AUTO, targetSize: int = 24) -> QCursor:
        curFile: QFile = QtCore.QFile(self.path(theme))
        if curFile.open(QtCore.QIODevice.ReadOnly):
            pixmap = QPixmap.fromImage(
                QImage.fromData(curFile.readAll(), b'ICO'))
            if not pixmap.isNull():
                curFile.seek(10)
                stream = QtCore.QDataStream(curFile)
                stream.setByteOrder(QtCore.QDataStream.LittleEndian)
                hotSpotX = int(stream.readUInt16() * targetSize / pixmap.width())
                hotSpotY = int(stream.readUInt16() * targetSize / pixmap.height())
                return QCursor(
                    pixmap.scaled(targetSize, targetSize, Qt.KeepAspectRatio, Qt.SmoothTransformation),
                    hotSpotX,
                    hotSpotY
                )
        return QCursor(QPixmap(self.path(theme)).scaled(
            targetSize,
            targetSize,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
