import inspect
import math
import random
import os.path
from pathlib import Path
from urllib.parse import urlparse

import PIL.Image
import requests
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QColor, QCursor, QPixmap
from PyQt5.QtWidgets import QWidget, QFileDialog
from numpy import ndarray
from qfluentwidgets import ToolTipFilter, ToolTipPosition, TransparentPushButton, Icon

from common.config import cfg, FORMATTED_IMG_SUFFIX
from common.icon import CursorIcon


class Util:

    @staticmethod
    def getRandomPic() -> str:
        height: int = random.randint(3, 6) * 100
        width: int = random.randint(3, 6) * 100
        fileName: str = f"{width}x{height}.png"
        filePath: str = f"resource/image/placeholder/{fileName}"
        if not os.path.exists(filePath):
            url: str = f"https://placehold.co/{fileName}"
            filePath = Util.downloadImage(url, filePath)
        return filePath

    @staticmethod
    def downloadImage(url: str, targetFilePath: str = None):
        print(f'Downloading image from {url}')
        response: requests.Response = requests.get(url)
        targetFilePath = \
            f'{cfg.cacheFolder.value}/{os.path.basename(urlparse(url).path)}.{FORMATTED_IMG_SUFFIX}' \
            if targetFilePath is None \
            else targetFilePath
        print(f'Saving image to {targetFilePath}')
        with open(targetFilePath, "wb") as file:
            file.write(response.content)
        return targetFilePath

    @staticmethod
    def retrieveName(var) -> str:
        callersLocalVars = inspect.currentframe().f_back.f_locals.items()
        return [varName for varName, varVal in callersLocalVars if varVal is var][0]

    @staticmethod
    def setModernTooltipText(qWidget: QWidget, text: str):
        qWidget.setToolTip(text)
        qWidget.installEventFilter(
            ToolTipFilter(qWidget, 0, ToolTipPosition.TOP))

    @staticmethod
    def fixTPBStyle(tpb: TransparentPushButton):
        tpb.setStyleSheet(
            tpb.styleSheet() +
            '''
            TransparentPushButton:disabled {
               background-color: transparent;
               border: none;
            }
            '''
        )

    @staticmethod
    def getOpenFileName(parent: QWidget = None) -> str:
        filePath, fileType = QFileDialog.getOpenFileName(
            parent,
            "选取文件",
            os.getcwd(),  # 起始路径
            "图片(*.gif;*.jpg;*.jpeg;*.bmp;*.jfif;*.png;*.svg;*.ico);;"
            "视频(*.mp4;*.mpeg;*.avi;*.mov;*.wmv;*.3gp;*.rmvb;*.flv;*.mkv)"
        )

        return filePath

    def getExistingDirectory(parent: QWidget = None) -> str:
        return QFileDialog.getExistingDirectory(
            parent,
            "选择文件保存位置",
            os.getcwd(),  # 起始路径
        )

    @staticmethod
    def copyWith(color: QColor, alpha: int = 255) -> QColor:
        color.setAlpha(alpha)
        return color

    @staticmethod
    def setCursor(widget: QWidget, cursorIcon: CursorIcon):
        widget.setCursor(cursorIcon.create())

    @staticmethod
    def expandMaskEdge(image: ndarray):
        edgePixels: list[tuple[int, int]] = []

        width: int = image.shape[0]
        height: int = image.shape[1]

        for x in range(width):
            for y in range(height):
                if image[x][y] == True and ((x - 1 >= 0 and image[x - 1][y] == False) or
                                            (x + 1 < width and image[x + 1][y] == False) or
                                            (y - 1 >= 0 and image[x][y - 1] == False) or
                                            (y + 1 < height and image[x][y + 1] == False)):
                    edgePixels.append((x, y))

        expandPixelSize: int = 10

        for pixel in edgePixels:
            for x in range(pixel[0] - expandPixelSize, pixel[0] + expandPixelSize + 1):
                for y in range(pixel[1] - expandPixelSize, pixel[1] + expandPixelSize + 1):
                    if 0 <= x < width and 0 <= y < height:
                        radius: float = math.sqrt(
                            (x - pixel[0]) ** 2 + (y - pixel[1]) ** 2)
                        if radius <= expandPixelSize:
                            image[x][y] = True

    @staticmethod
    def rectMask(image: ndarray):
        whitePixels: list[tuple[int, int]] = []

        width: int = image.shape[0]
        height: int = image.shape[1]

        top: int = -1
        bottom: int = -1
        left: int = -1
        right: int = -1

        for x in range(width):
            for y in range(height):
                if image[x][y] == True:
                    whitePixels.append((x, y))
                    if top == -1 or y < top:
                        top = y
                    if bottom == -1 or y > bottom:
                        bottom = y
                    if left == -1 or x < left:
                        left = x
                    if right == -1 or x > right:
                        right = x

        for x in range(left, right + 1):
            for y in range(top, bottom + 1):
                image[x][y] = True
