import string
import threading
import time

import PIL.Image
import PIL.Image
import PIL.Image
import PIL.Image
import PIL.Image
import PIL.Image
import PIL.ImageOps
import numpy
import numpy as np
from PIL import ImageDraw, Image
from PIL.Image import Image
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QEvent, QPointF, QRectF
from PyQt5.QtGui import QImage, QPainter, QColor
from PyQt5.QtWidgets import QGridLayout, QAbstractScrollArea, QScrollArea
from pyinpaint import Inpaint
from qfluentwidgets import SimpleCardWidget, ImageLabel
from qfluentwidgets import SmoothScrollDelegate
from rembg import remove
import skimage
from skimage.util import img_as_ubyte

from common.command import Command, CommandMode
from common.config import cfg, FORMATTED_IMG_SUFFIX
from common.icon import CursorIcon
from common.quick_action import QuickAction, QuickActionMode
from common.util import Util

from lib.simple_lama_inpainting.model import SimpleLama
from yolov7_package import Yolov7Detector
from segment_anything import SamPredictor
from deoldify.visualize import *
from skimage.metrics import structural_similarity

import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ScrollArea(QScrollArea):
    """
    滚动区域组件

    该组件重写了 qfluentwidgets 中的 ScrollArea 组件中的 scrollDelegate 属性，

    但请注意，您需要在子组件内部实现横向、纵向滚动的处理事件。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scrollDelegate = MySmoothScrollDelegate(self)


class MySmoothScrollDelegate(SmoothScrollDelegate):
    '''
    该组件继承自 SmoothScrollDelegate，重写了 eventFilter 方法，使其可以穿透到外部组件。
    '''

    def __init__(self, parent: QAbstractScrollArea, useAni=False):
        super().__init__(parent, useAni)

    def eventFilter(self, obj, e: QEvent):
        e.setAccepted(False)
        return False


class DraggableImage(ScrollArea):
    # 信号
    setZoomFactorLabelTextSignal: pyqtSignal = pyqtSignal()
    setRedoBtnEnabledSignal: pyqtSignal = pyqtSignal(bool)
    setUndoBtnEnabledSignal: pyqtSignal = pyqtSignal(bool)
    setSpinnerEnabledSignal: pyqtSignal = pyqtSignal(bool)
    setColorLabelTextSignal: pyqtSignal = pyqtSignal()

    # 图片的快照位于历史记录中的位置
    curSnapshotIndex: int = -1
    # 快照列表，维护一个图片修改历史记录。
    # 当图片被修改时，当前显示的图片应该被插入到该列表。
    _history: List[PIL.Image] = list()

    # 文件
    # Root directory of the project
    ROOT_DIR = os.path.abspath("../")
    fileName: str = ''
    imagePath: str = ''

    # 命令
    _persistentCursor: CursorIcon = CursorIcon.POINTER
    _command: Command = Command(CommandMode.UNKNOWN)
    _commandEnabled: bool = False

    # 画笔
    _painter: QPainter = QPainter()
    _drawingImage: QImage = None
    _maskToInpaint: PIL.Image = None
    penColor: QColor = QColor(Qt.black)
    penWidthFactor: int = 2
    _rectSelectPenWidthFactor: int = 1
    _pointSelectPenWidthFactor: int = 10
    _paintSelectPenWidthFactor: int = 20
    _cropPenWidthFactor: int = 1

    _lastPos: QPoint
    _startPos: QPoint
    _points: list[QPoint] = list()

    # 移动
    _isMoveMode: bool = False
    isCtrlKeyPressed: bool = False
    _isLeftBtnPressed: bool = False
    _isRightBtnPressed: bool = False

    # 缩放
    zoomStep: float = 0.05
    zoomFactor: float = 1.0

    imageColorizer: ModelImageVisualizer
    simpleLama: SimpleLama
    yolov7Detector: Yolov7Detector
    samPredictor: SamPredictor

    def __init__(self, imageColorizer: ModelImageVisualizer, simpleLama: SimpleLama, maskGenerator: SamPredictor, yolov7Detector: Yolov7Detector, parent=None):
        super().__init__(parent)

        self.imageColorizer = imageColorizer
        self.simpleLama = simpleLama
        self.samPredictor = maskGenerator
        self.yolov7Detector = yolov7Detector

        self.imageLabel: ImageLabel = ImageLabel()

        imageLayout: QGridLayout = QGridLayout()
        imageLayout.addWidget(self.imageLabel, 0, 0, 1, 1)
        imageLayout.setContentsMargins(0, 0, 0, 0)

        imageWidget: SimpleCardWidget = SimpleCardWidget()
        imageWidget.setStyleSheet('background-color: transparent;')
        imageWidget.setLayout(imageLayout)

        self.setWidget(imageWidget)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setViewportMargins(0, 0, 8, 8)

    def initImage(self, path: str, fileName: str):
        '''
        初始化图片区域内容
        :param path: 图片路径
        :param fileName: 文件名（包含文件格式后缀）
        :return:
        '''

        self.setSpinnerEnabledSignal.emit(True)

        def task():
            print(f'Opening file: {path}')
            self.fileName = f"{fileName[0:fileName.rindex('.')]}.{FORMATTED_IMG_SUFFIX}"

            # self.originalImagePath: str = Util.getRandomPic()
            self.imagePath = f'{cfg.workFolder.value}/{self.fileName}'
            if path != self.imagePath:
                print(f'Converting file to .{FORMATTED_IMG_SUFFIX} format')
                Image.open(path).save(self.imagePath)
            self.updateHistory(Image.open(self.imagePath))
            self.getEntropy(self.getCurrentSnapshot())
            self.showCurrentSnapshotToUI()
            self.onQuickActionBarClicked(
                QuickAction(QuickActionMode.SCALE_TO_WINDOW))

            self.setSpinnerEnabledSignal.emit(False)

        threading.Thread(target=task).start()

    def getPenWidth(self, penWidthFactor: int = None) -> float:
        '''
        获取画笔宽度
        :param penWidthFactor: 画笔宽度因子
        :return: 返回计算后的画笔宽度
        '''
        return penWidthFactor / self.zoomFactor

    def onQuickActionBarClicked(self, quickAction: QuickAction, arg: Any = None):
        '''
        快速命令按钮点按响应事件
        :param quickAction: 快速命令
        :param arg: 可选参数，默认为 None
        :return:
        '''

        if quickAction.takeTime:
            self.setSpinnerEnabledSignal.emit(True)

        mode: QuickActionMode = quickAction.mode

        def task():
            print(f'Action `{mode}` performed')
            if mode == QuickActionMode.REMOVE_BACKGROUND:
                self.removeBg()
            elif mode == QuickActionMode.TEXT:
                self.removeByPrompt(arg)
            elif mode == QuickActionMode.BRUSH:
                self.colorizeImage()
            elif mode == QuickActionMode.ROTATE:
                self.rotate()
            elif mode == QuickActionMode.ZOOM_IN:
                self.zoom(True)
            elif mode == QuickActionMode.ZOOM_OUT:
                self.zoom(False)
            elif mode == QuickActionMode.SCALE_TO_RAW:
                self.scaleToRaw()
            elif mode == QuickActionMode.SCALE_TO_WINDOW:
                self.scaleToWindow()
            elif mode == QuickActionMode.COLOR_OVERLAY:
                self.overlayColor()
            elif mode == QuickActionMode.IMAGE_OVERLAY:
                self.overlayImage(arg)
            elif mode == QuickActionMode.PURIFY_BACKGROUND:
                self.purifyBg()
            elif mode == QuickActionMode.UNDO:
                self.undo()
            elif mode == QuickActionMode.REDO:
                self.redo()
            self.updateRedoUndoState()

            if quickAction.takeTime:
                self.setSpinnerEnabledSignal.emit(False)

        threading.Thread(target=task).start()

    def onCommandClicked(self, command: Command):
        '''
        响应命令按钮的点击事件
        :param mode: 命令模式
        :return:
        '''

        mode: CommandMode = command.mode

        print(f'Command `{mode}` activated')

        if mode == CommandMode.RECTANGLE_SELECT_REMOVE:
            self._persistentCursor = CursorIcon.PRECISION
        elif mode == CommandMode.POINT_SELECT_REMOVE:
            # self.genMaskForPointRemove()
            # Firstly, it automatically generates mask to let user have a better
            # way to choose which parts are expected to be inpainted,
            # but it can be too slow (~4min),
            # so I leave it commented and wait to be solved in the future.
            self._persistentCursor = CursorIcon.PRECISION
        elif mode == CommandMode.PAINT_SELECT_REMOVE:
            self._persistentCursor = CursorIcon.HANDWRITING
        elif mode == CommandMode.DRAW:
            self._persistentCursor = CursorIcon.HANDWRITING
        elif mode == CommandMode.PAINT_BUCKET:
            self._persistentCursor = CursorIcon.PRECISION
        elif mode == CommandMode.DROPPER:
            self._persistentCursor = CursorIcon.PRECISION
        elif mode == CommandMode.CROP:
            self._persistentCursor = CursorIcon.PRECISION
        elif mode == CommandMode.UNKNOWN:
            self._persistentCursor = CursorIcon.POINTER
        self._command = command
        self.changeCursorIcon()

    def getAnns(self, anns: list[dict[str, Any]]) -> Image:
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

        img = np.ones(
            (
                sorted_anns[0]['segmentation'].shape[0],
                sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        return Image.fromarray(img_as_ubyte(img))

    # def genMaskForPointRemove(self):
    #     self.setSpinnerEnabledSignal.emit(True)

    #     def task():
    #         self.initAutoMaskGenerator()
    #         masks = self.autoMaskGenerator.generate(
    #             numpy.array(self.getCurrentSnapshot()))
    #         self.updateHistory(self.getAnns(masks))
    #         self.showCurrentSnapshotToUI()
    #         self.setSpinnerEnabledSignal.emit(False)

    #     threading.Thread(target=task).start()

    def changeCursorIcon(self, cursorIcon: CursorIcon = None):
        Util.setCursor(
            self.imageLabel, self._persistentCursor if cursorIcon is None else cursorIcon)

    def updateRedoUndoState(self):
        '''
        更新撤销重做按钮的状态
        :return:
        '''
        self.setRedoBtnEnabledSignal.emit(self.redoable())
        self.setUndoBtnEnabledSignal.emit(self.undoable())

    def showCurrentSnapshotToUI(self):
        '''
        将当前快照（快照历史记录中的当前项，并非当前显示在 UI 上的图像）显示到 UI 上。

        更新当前展示的图片前，将快照内的图片缩放至当前的缩放比例。
        '''
        self.imageLabel.setImage(self.scaleToZoomFactor(
            self.getCurrentSnapshot().toqimage(), self.zoomFactor))

    def showDrawingToUI(self):
        '''
        将当前使用画笔等工具绘制的图片（即 self._drawingImage）显示到 UI 上

        此操作不会更新快照列表
        :return:
        '''
        self.imageLabel.setImage(self.scaleToZoomFactor(
            self._drawingImage, self.zoomFactor))

    def undoable(self) -> bool:
        '''
        是否可以撤销
        :return: 返回一个值，指示知否可以撤销
        '''
        return self.curSnapshotIndex > 0

    def undo(self):
        '''
        撤销
        :return:
        '''
        if self.undoable():
            self.curSnapshotIndex -= 1
            self.showCurrentSnapshotToUI()

    def redoable(self) -> bool:
        '''
        是否可以重做
        :return: 返回一个值，指示是否可以重做
        '''
        return self.curSnapshotIndex + 1 < self._history.__len__()

    def redo(self):
        '''
        重做
        :return:
        '''
        if self.redoable():
            self.curSnapshotIndex += 1
            self.showCurrentSnapshotToUI()

    def updateHistory(self, image: PIL.Image):
        '''
        更新快照列表
        :param image:需要加入至快照列表的图片
        :return:
        '''
        # 当前显示图片对应的快照不在历史记录列表的最后一一位
        if self.curSnapshotIndex < self._history.__len__() - 1:
            self._history = self._history[0:self.curSnapshotIndex + 1]
        self.curSnapshotIndex += 1
        self._history.append(image)

    def rotate(self):
        '''
        旋转
        :return:
        '''
        # Crop while rotating image
        # Problem fixed: https://stackoverflow.com/a/67138829/11048731
        self.updateHistory(self.getCurrentSnapshot().rotate(90, expand=1))
        self.showCurrentSnapshotToUI()

    # 快照
    def getCurrentSnapshot(self) -> PIL.Image:
        '''
        获取当前快照
        :return:
        '''
        return self._history[self.curSnapshotIndex]

    def saveCurrentSnapshotToCacheFolder(self) -> str:
        '''
        保存当前快照至缓存区
        :return: 返回一个值，指示保存文件的地址
        '''
        cacheFilePath: str = f'{cfg.cacheFolder.value}/{self.fileName}'
        with open(cacheFilePath, 'wb'):
            self.getCurrentSnapshot().save(cacheFilePath)
        return cacheFilePath

    def saveImageToCacheFolder(self, img: PIL.Image) -> str:
        '''
        保存图像至缓存区
        :return: 返回一个值，指示保存文件的地址
        '''
        s = string.ascii_lowercase+string.digits
        randomFileName = ''.join(random.sample(s, 10))
        cacheFilePath: str = f'{cfg.cacheFolder.value}/{randomFileName}.png'
        with open(cacheFilePath, 'wb'):
            img.save(cacheFilePath)
        return cacheFilePath

    def saveCurrentSnapshotToWorkFolder(self) -> bool:
        '''
        保存当前快照至工作区
        :return: 返回一个值，指示是否保存成功
        '''
        workFilePath: str = f'{cfg.workFolder.value}/{self.fileName}'
        with open(workFilePath, 'wb'):
            return self.getCurrentSnapshot().save(workFilePath)

    def saveComparedImages(self, dirPath: str):
        '''
        保存对比图片
        :return:
        '''
        firstImg: PIL.Image = self._history[0]
        currentImg: PIL.Image = self.getCurrentSnapshot()
        if self._maskToInpaint is None:
            newImg: PIL.Image = Image.new(
                'RGBA', (firstImg.width * 2, firstImg.height))
            newImg.paste(firstImg, (0, 0))
            newImg.paste(currentImg, (firstImg.width, 0))
            newImg.save(f'{dirPath}/{self.fileName}')
        else:
            newImg: PIL.Image = Image.new(
                'RGBA', (firstImg.width * 3, firstImg.height))
            newImg.paste(firstImg, (0, 0))
            newImg.paste(self._maskToInpaint, (firstImg.width, 0))
            newImg.paste(currentImg, (firstImg.width * 2, 0))
            newImg.save(f'{dirPath}/{self.fileName}')

    def deleteStorageFile(self):
        '''
        删除本地工作区的文件
        :return:
        '''
        pathlib.Path.unlink(pathlib.Path(self.imagePath))

    def inpaint(self, orig: PIL.Image, mask: PIL.Image, useLama: bool = True) -> PIL.Image:
        if useLama:
            print('Inpainting using LaMa...')
            start = time.time()
            res: PIL.Image = self.simpleLama(orig.convert('RGB'), mask)
            end = time.time()
            print(f'Inpainting finished in {end - start} seconds')
            return res
        else:
            inpaint = Inpaint(
                org_img=self.saveImageToCacheFolder(orig),
                mask=self.saveImageToCacheFolder(mask),
                ps=3)
            return Image.fromarray(inpaint(k_boundary=16, k_search=300, k_patch=3))

    def rmBgUsingU2Net(self, origImg: PIL.Image, onlyMask: bool = True) -> PIL.Image:
        additionalMsg: str = '(and generating mask) ' if onlyMask else ''
        print(f'Removing background {additionalMsg}using U2Net...')
        t = time.time()
        mask: PIL.Image = remove(origImg, only_mask=onlyMask)
        print(
            f'Background removed {additionalMsg}in {time.time() - t} seconds')
        return mask

    def generateMaskBySam(
            self,
            origImg: PIL.Image,
            point_coords: numpy.ndarray | None = None,
            point_labels: numpy.ndarray | None = None,
            box: numpy.ndarray | None = None) -> PIL.Image:
        image = numpy.array(origImg)
        print(f'Generating mask using SAM...')
        start = time.time()
        self.samPredictor.set_image(image)
        masks, _, _ = self.samPredictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=False
        )
        end = time.time()
        print(f'Mask generated in {end - start} seconds')
        mask = masks[0]
        print('Expanding mask edge...')
        start = time.time()
        Util.expandMaskEdge(mask)
        end = time.time()
        print(f'Mask edge expanded in {end - start} seconds')
        return Image.fromarray(mask)

    def getEntropy(self, pilImg: PIL.Image) -> float:
        entropy: float = skimage.measure.shannon_entropy(np.array(pilImg.convert('L')))
        print(f'Entropy of image: {entropy}')
        print(f'Pixel count: {pilImg.width * pilImg.height}')
        return entropy

    def overlayColor(self):
        '''
        将图片透明区域给予颜色填充
        :return:
        '''
        img: PIL.Image = self.getCurrentSnapshot().convert('RGBA')
        colorToFill: tuple[int, int, int, int] = \
            (self.penColor.red(), self.penColor.green(), self.penColor.blue(), 255)
        bgImg: PIL.Image = Image.new('RGBA', img.size, colorToFill)
        bgImg.paste(img, mask=img)
        self.updateHistory(bgImg)
        self.showCurrentSnapshotToUI()

    def overlayImage(self, path: str):
        '''
        将图片透明区域给予图片填充
        :param path:
        :return:
        '''
        if path != '':
            img: PIL.Image = self.getCurrentSnapshot()
            bgImg: PIL.Image = Image.open(path).convert('RGBA')
            bgImgRatio: float = bgImg.width / bgImg.height
            if int(img.height * bgImgRatio) > img.width:
                bgImg = bgImg.resize(
                    (int(img.height * bgImgRatio), img.height))
                offset: int = int((bgImg.width - img.width) / 2)
                bgImg = bgImg.crop((offset, 0, offset + img.width, img.height))
            else:
                bgImg = bgImg.resize((img.width, int(img.width / bgImgRatio)))
                offset: int = int((bgImg.height - img.height) / 2)
                bgImg = bgImg.crop((0, offset, img.width, offset + img.height))
            bgImg.paste(img, mask=img)
            self.updateHistory(bgImg)
            self.showCurrentSnapshotToUI()

    def purifyBg(self):
        origImg: PIL.Image = self.getCurrentSnapshot()
        mask: PIL.Image = self.rmBgUsingU2Net(origImg, True)
        self._maskToInpaint = mask
        self.updateHistory(self.inpaint(origImg, self._maskToInpaint))
        self.showCurrentSnapshotToUI()

    def colorizeImage(self):
        '''
        图片色彩化，使用 DeOldify
        :return:
        '''

        origImg: PIL.Image = self.getCurrentSnapshot()

        X = []

        # 修正中文显示错误
        plt.rcParams['font.sans-serif'] = ['Simhei']
        # 在后台运行
        matplotlib.use('agg')
        plt.xlabel('渲染因子')
        plt.ylabel('SSIM')

        for factor in range(8, 42, 2):
            img: PIL.Image = self.imageColorizer.get_transformed_image(
                path=self.saveCurrentSnapshotToCacheFolder(),
                render_factor=factor, # 默认 35
                watermarked=False)
            ssim: float = structural_similarity(
                np.array(origImg), np.array(img), channel_axis=2)
            X.append([ssim, factor])
            print(f'SSIM: {ssim} at factor {factor}')

        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(np.array(X))

        # 计算不同聚类数的SSE
        inertia = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)

        # 绘制手肘图
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
        plt.xlabel('聚类数')
        plt.ylabel('SSE')
        plt.grid(True)
        plt.savefig('myplot.png')

        # 人工观察
        bestCluster = 2

        kmeans = KMeans(n_clusters=bestCluster, random_state=42)
        kmeans.fit(X)

        # 获取质心
        centroids = kmeans.cluster_centers_
        bestFactor = 33 # centroids[bestCluster - 1][1]
        print(f'Best SSIM founded with {centroids[bestCluster - 1][0]} at factor {centroids[bestCluster - 1][1]}')

        bestImg = self.imageColorizer.get_transformed_image(
                path=self.saveCurrentSnapshotToCacheFolder(),
                render_factor=bestFactor,
                watermarked=False)
        self.updateHistory(bestImg)
        self.showCurrentSnapshotToUI()
        self.updateRedoUndoState()

    # def colorizeVideo(self):
    #     '''
    #     视频上色，使用 DeOldify
    #     :return:
    #     '''
    #     # self.videoColorizer
    #     pass

    def removeBg(self):
        '''
        移除背景
        '''

        self.updateHistory(self.rmBgUsingU2Net(
            self.getCurrentSnapshot(), False))
        self.showCurrentSnapshotToUI()
        self.updateRedoUndoState()

    def removeByPrompt(self, prompt: str):
        '''
        通过提示移除背景
        :param prompt: 提示
        :return:
        '''
        origImg: PIL.Image = self.getCurrentSnapshot()
        imageArr = np.array(origImg)
        classes, boxes, scores = self.yolov7Detector.detect(imageArr)
        self.samPredictor.set_image(imageArr)
        inputBoxes = torch.tensor(boxes)
        transformedBoxes = self.samPredictor.transform.apply_boxes_torch(
            inputBoxes, imageArr.shape[:2])
        masks, _, _ = self.samPredictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformedBoxes,
            multimask_output=False
        )
        combinedMask = np.zeros(
            (imageArr.shape[0], imageArr.shape[1]), dtype=np.uint8)
        for batchIndex in range(0, len(classes[0])):
            classId = classes[0][batchIndex]
            className = self.yolov7Detector.names[classId]
            if className in prompt:
                cur = masks[batchIndex][0].cpu().numpy()
                Util.expandMaskEdge(cur)
                combinedMask = np.logical_or(combinedMask, cur)
        self._maskToInpaint = Image.fromarray(combinedMask)
        imageToUpdate = self.inpaint(origImg, self._maskToInpaint)

        self.updateHistory(imageToUpdate)
        self.showCurrentSnapshotToUI()

    def zoom(self, zoomIn: bool):
        '''
        缩放当前显示在 UI 上的图片
        :param zoomIn: 是否放大，True 则放大，False 则缩小
        :return:
        '''
        zoomFactorAfter: float = self.zoomFactor + \
            self.zoomStep * (1 if zoomIn > 0 else -1)
        if zoomFactorAfter > 0:
            self.zoomFactor = zoomFactorAfter
            self.setZoomFactorLabelTextSignal.emit()
        self.showCurrentSnapshotToUI()

    def scaleToZoomFactor(self, qImage: QImage, zoomFactor: float) -> QImage:
        '''
        将图片缩放至当前缩放因子大小
        :param qImage: 需要处理的图片
        :param zoomFactor: 缩放因子
        :return:
        '''
        return qImage.smoothScaled(int(qImage.width() * zoomFactor), int(qImage.height() * zoomFactor))

    def scaleToWindow(self):
        '''
        将图片缩放至自适应窗口大小
        :return:
        '''
        originalWidth: int = self.getCurrentSnapshot().size[0]
        ratio: float = originalWidth / self.getCurrentSnapshot().size[1]
        viewPortWidth: int = self.viewport().width()
        targetWidth: int = int(self.viewport().height() * ratio)
        if targetWidth > viewPortWidth:
            targetWidth = viewPortWidth
        self.zoomFactor = targetWidth / originalWidth
        self.showCurrentSnapshotToUI()
        self.setZoomFactorLabelTextSignal.emit()

    def scaleToRaw(self):
        '''
        将图片缩放至原始大小
        :return:
        '''
        self.zoomFactor = 1
        self.showCurrentSnapshotToUI()
        self.setZoomFactorLabelTextSignal.emit()

    def dropper(self, point: QPointF):
        self.penColor = QColor.fromRgb(
            *self.getCurrentSnapshot().getpixel((int(point.x()), int(point.y()))))

    def mapToActualPoint(self, targetPos: QPointF) -> QPointF:
        '''
        将目标点映射到原始尺寸的图片上，并返回这个模拟的点
        :param targetPos: 需要处理的点
        :return: 映射后的点
        '''
        horizontalMargin: float = (
            self.viewport().width() - self.imageLabel.image.width()) / 2
        verticalMargin: float = (
            self.viewport().height() - self.imageLabel.image.height()) / 2
        qPointMoveDelta: QPointF = QPointF(
            -horizontalMargin if horizontalMargin > 0 else self.horizontalScrollBar().value(),
            -verticalMargin if verticalMargin > 0 else self.verticalScrollBar().value()
        )
        return (targetPos + qPointMoveDelta) / self.zoomFactor

    def rebootPainter(
            self,
            penStyle: Qt.PenStyle = Qt.SolidLine,
            penColor: QColor = None,
            penWidthFactor: int = None):
        """
        重启画笔，将画笔绘制对象设置为当前快照的复制品
        :param penStyle: 画笔样式
        :param penColor: 画笔颜色，None 则指定其值为 self.penColor
        :param penWidthFactor: 画笔宽度因子，None 则指定其值为 self.penWidthFactor
        :return:
        """
        if self._painter.isActive():
            self._painter.end()
        if penColor is None:
            penColor = self.penColor
        if penWidthFactor is None:
            penWidthFactor = self.penWidthFactor
        penWidth: float = self.getPenWidth(penWidthFactor)
        self._drawingImage = self.getCurrentSnapshot().toqimage().copy()
        self._painter.begin(self._drawingImage)
        self._painter.setRenderHint(QPainter.Antialiasing, True)
        self._painter.setPen(QtGui.QPen(
            penColor,
            penWidth,
            penStyle,
            Qt.RoundCap,
            Qt.RoundJoin
        ))

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key_Control:
            self.isCtrlKeyPressed = True

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key_Control:
            self.isCtrlKeyPressed = False

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        # 判断光标是否位于图像内
        if self.imageLabel.geometry().contains(event.pos()):

            self._lastPos = event.pos()
            self._startPos = event.pos()

            if event.button() == Qt.RightButton:
                if self._command.mode == CommandMode.POINT_SELECT_REMOVE:
                    if self._isLeftBtnPressed:
                        self.rebootPainter(
                            penColor=QColor(Qt.red),
                            penWidthFactor=self._pointSelectPenWidthFactor)
                        self._points.append(event.pos())
                        mappedPoints: list[QPointF] = list(
                            map(lambda point: self.mapToActualPoint(point), self._points))
                        self._painter.drawPoints(mappedPoints)
                        self.showDrawingToUI()
                elif self._command.mode == CommandMode.PAINT_SELECT_REMOVE:
                    if self._isLeftBtnPressed:
                        self._points.clear()
                        self._isRightBtnPressed = True
                else:
                    self._isMoveMode = True
                    self.changeCursorIcon(CursorIcon.MOVE)

            elif event.button() == Qt.LeftButton:
                self._points.clear()
                self._isLeftBtnPressed = True
                if self._command.mode == CommandMode.DRAW:
                    self.rebootPainter()
                elif self._command.mode == CommandMode.PAINT_SELECT_REMOVE:
                    self.rebootPainter(
                        penColor=QColor(Qt.black),
                        penWidthFactor=self._paintSelectPenWidthFactor)
                    self._maskToInpaint = PIL.Image.new('L', (
                        self.getCurrentSnapshot().size[0],
                        self.getCurrentSnapshot().size[1]), 0)
                elif self._command.mode == CommandMode.DROPPER:
                    self.dropper(self.mapToActualPoint(self._startPos))

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.imageLabel.geometry().contains(event.pos()):

            curPos: QPoint = event.pos()
            mappedCurPos: QPointF = self.mapToActualPoint(curPos)
            mappedStartPos: QPointF = self.mapToActualPoint(self._startPos)
            mappedLastPos: QPointF = self.mapToActualPoint(self._lastPos)

            self.setColorLabelTextSignal.emit()

            if self._isMoveMode:
                delta: QPoint = curPos - self._lastPos
                self.horizontalScrollBar().setValue(
                    self.horizontalScrollBar().value() - delta.x())
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())

            elif self._command.mode == CommandMode.DRAW:
                self._painter.drawLine(mappedLastPos, mappedCurPos)
                self.showDrawingToUI()

            elif self._command.mode == CommandMode.CROP:
                self.rebootPainter(
                    penStyle=Qt.DashLine,
                    penWidthFactor=self._cropPenWidthFactor)
                self._painter.drawRect(QRectF(mappedStartPos, mappedCurPos))
                self.showDrawingToUI()

            elif self._command.mode == CommandMode.RECTANGLE_SELECT_REMOVE:
                self.rebootPainter(
                    penColor=QColor(Qt.black),
                    penWidthFactor=self._rectSelectPenWidthFactor)
                self._painter.drawRect(QRectF(mappedStartPos, mappedCurPos))
                self.showDrawingToUI()

            elif self._command.mode == CommandMode.PAINT_SELECT_REMOVE:
                if self._isLeftBtnPressed and self._isRightBtnPressed:
                    self._painter.drawLine(mappedLastPos, mappedCurPos)
                    self.showDrawingToUI()
                    self._points.append(curPos)

            elif self._command.mode == CommandMode.DROPPER:
                self.dropper(self.mapToActualPoint(curPos))

            self._lastPos = curPos

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.imageLabel.geometry().contains(event.pos()):

            if event.button() == Qt.LeftButton:
                self._isLeftBtnPressed = False
            if event.button() == Qt.RightButton:
                self._isRightBtnPressed = False

            endPoint = event.pos()

            mappedStartPos: QPointF = self.mapToActualPoint(self._startPos)
            mappedEndPos: QPointF = self.mapToActualPoint(endPoint)

            mappedCenterPos: QPointF = (mappedStartPos + mappedEndPos) / 2
            offset: QPointF = (mappedEndPos - mappedStartPos) / 2
            mappedTopLeftPos: QPointF = QPointF(mappedCenterPos.x() - offset.x().__abs__(),
                                                mappedCenterPos.y() - offset.y().__abs__())
            mappedBottomRightPos: QPointF = QPointF(mappedCenterPos.x() + offset.x().__abs__(),
                                                    mappedCenterPos.y() + offset.y().__abs__())

            mappedPoints: list[QPointF] = list(
                map(lambda point: self.mapToActualPoint(point), self._points))

            origImg: PIL.Image = self.getCurrentSnapshot()

            self.changeCursorIcon()

            def task():
                if self._command.takeTime:
                    self.setSpinnerEnabledSignal.emit(True)

                imageToUpdate: PIL.Image = None

                if self._isMoveMode:
                    self._isMoveMode = False

                elif self._command.mode == CommandMode.CROP:
                    imageToUpdate = origImg.crop((
                        mappedTopLeftPos.x(),
                        mappedTopLeftPos.y(),
                        mappedBottomRightPos.x(),
                        mappedBottomRightPos.y()))

                elif self._command.mode == CommandMode.RECTANGLE_SELECT_REMOVE:
                    inputBox = np.array([
                        mappedTopLeftPos.x(),
                        mappedTopLeftPos.y(),
                        mappedBottomRightPos.x(),
                        mappedBottomRightPos.y()])
                    mask = self.generateMaskBySam(
                        origImg=origImg,
                        box=inputBox[None, :]
                    )
                    self._maskToInpaint = mask
                    imageToUpdate = self.inpaint(origImg, self._maskToInpaint)

                elif self._command.mode == CommandMode.POINT_SELECT_REMOVE:
                    if not self._isLeftBtnPressed:
                        inputPoint = np.array(
                            [[pos.x(), pos.y()] for pos in mappedPoints])
                        inputLabel = np.array([1 for _ in mappedPoints])
                        mask = self.generateMaskBySam(
                            origImg=origImg,
                            point_coords=inputPoint,
                            point_labels=inputLabel
                        )
                        self._maskToInpaint = mask
                        imageToUpdate = self.inpaint(
                            origImg, self._maskToInpaint)

                elif self._command.mode == CommandMode.PAINT_SELECT_REMOVE:
                    if self._isLeftBtnPressed:
                        ImageDraw.Draw(self._maskToInpaint).line(
                            xy=[(point.x(), point.y())
                                for point in mappedPoints],
                            fill=255,
                            width=int(self.getPenWidth(
                                self._paintSelectPenWidthFactor)),
                            joint="curve"
                        )
                    else:
                        imageToUpdate = self.inpaint(
                            origImg, self._maskToInpaint)

                elif self._command.mode == CommandMode.DRAW:
                    imageToUpdate = Image.fromqimage(self._drawingImage)

                elif self._command.mode == CommandMode.PAINT_BUCKET:
                    seedPos: QPoint = self.mapToActualPoint(self._lastPos)
                    img: PIL.Image = self.getCurrentSnapshot().convert('RGBA')
                    ImageDraw.floodfill(img,
                                        (seedPos.x(), seedPos.y()),
                                        self.penColor.getRgb(),
                                        thresh=40)
                    imageToUpdate = img

                if imageToUpdate is not None:
                    self.updateHistory(imageToUpdate)
                    self.showCurrentSnapshotToUI()
                    self.updateRedoUndoState()

                if self._command.takeTime:
                    self.setSpinnerEnabledSignal.emit(False)

            threading.Thread(target=task).start()

    def wheelEvent(self, event: QtGui.QWheelEvent):
        if self.isCtrlKeyPressed:
            self.zoom(event.angleDelta().y() > 0)
        else:
            if event.angleDelta().y() != 0:
                if not self.scrollDelegate.useAni:
                    self.scrollDelegate.verticalSmoothScroll.wheelEvent(event)
                else:
                    self.scrollDelegate.vScrollBar.scrollValue(
                        -event.angleDelta().y())
            else:
                if not self.scrollDelegate.useAni:
                    self.scrollDelegate.horizonSmoothScroll.wheelEvent(event)
                else:
                    self.scrollDelegate.hScrollBar.scrollValue(
                        -event.angleDelta().x())
