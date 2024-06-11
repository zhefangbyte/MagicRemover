# coding:utf-8
import os.path
import time

import torch.cuda
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QGridLayout, QWidget
from qfluentwidgets import FluentIcon as FIF, SplashScreen, MSFluentWindow, IndeterminateProgressRing
from qfluentwidgets import (NavigationItemPosition)

from common.config import FORMATTED_IMG_SUFFIX
from common.icon import CursorIcon
from common.util import Util
from view.home_interface import HomeInterface
from view.setting_interface import SettingInterface
from view.work_interface import WorkInterface
from lib.simple_lama_inpainting.model import SimpleLama
from yolov7_package import Yolov7Detector
from segment_anything import SamPredictor, sam_model_registry
from deoldify.visualize import *


class MainWindow(MSFluentWindow):
    workInterface: WorkInterface = None

    _spinnerMask: QWidget
    _spinner: IndeterminateProgressRing
    _spinnerStartTime: int = 0

    imageColorizer: ModelImageVisualizer
    simpleLama: SimpleLama
    yolov7Detector: Yolov7Detector
    maskGenerator: SamPredictor
    # videoColorizer: VideoColorizer = None
    # autoMaskGenerator: SamAutomaticMaskGenerator = None

    def __init__(self):
        super().__init__()

        print('[1/4]Loading models...')
        self.initImageColorizer()
        print('[2/4]Loading models...')
        self.initSimpleLama()
        print('[3/4]Loading models...')
        self.initYOLOv7()
        print('[4/4]Loading models...')
        self.initSegmentPredictor()
        print('Models loaded, starting GUI...')

        rootWidget: QWidget = QWidget()
        rootWidget.setLayout(self.hBoxLayout)

        self._spinner = IndeterminateProgressRing()
        self._spinnerMask = QWidget()
        self._spinnerMask.setStyleSheet(
            'background-color: rgba(0, 0, 0, 0.5);')
        self.hideSpinner()

        rootLayout: QGridLayout = QGridLayout(self)
        rootLayout.setContentsMargins(0, 0, 0, 0)
        rootLayout.addWidget(rootWidget, 0, 0, 1, 1)
        rootLayout.addWidget(self._spinnerMask, 0, 0, 1, 1)
        rootLayout.addWidget(self._spinner, 0, 0, 1, 1,
                             alignment=Qt.AlignCenter)

        self.titleBar.raise_()

        self.initWindow()
        self.initNavigation()
        self.splashScreen.finish()

    def initNavigation(self):
        homeInterfaceLabel = '主页'
        settingInterfaceLabel = '设置'

        self.homeInterface = HomeInterface(Util.retrieveName(
            homeInterfaceLabel), homeInterfaceLabel, self)
        self.homeInterface.openWorkspaceSignal.connect(self.openWorkspace)
        self.settingInterface = SettingInterface(
            Util.retrieveName(settingInterfaceLabel), settingInterfaceLabel, self)

        self.addSubInterface(self.homeInterface, FIF.HOME, homeInterfaceLabel)
        self.addSubInterface(self.settingInterface, FIF.SETTING, settingInterfaceLabel,
                             position=NavigationItemPosition.BOTTOM)

        Util.setCursor(self, CursorIcon.POINTER)

    def initWindow(self):
        self.resize(900, 700)
        self.setWindowIcon(QIcon('resource/image/logo.png'))
        self.setWindowTitle('MagicRemover')

        self.setMicaEffectEnabled(True)

        # create splash screen
        self.splashScreen = SplashScreen(self.windowIcon(), self)
        self.splashScreen.setIconSize(QSize(106, 106))
        self.splashScreen.raise_()

        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)
        self.show()

        QApplication.processEvents()

    def showSpinner(self):
        self._spinnerMask.show()
        self._spinner.show()
        self._spinner.start()

    def hideSpinner(self):
        self._spinner.stop()
        self._spinner.hide()
        self._spinnerMask.hide()

    def onSpinnerEnabledSet(self, value: bool):
        if value:
            self.showSpinner()
            self._spinnerStartTime = int(time.time())
        else:
            self.hideSpinner()
            endTime = time.time()
            min = int((endTime - self._spinnerStartTime) / 60)
            sec = int((endTime - self._spinnerStartTime) % 60)
            print(f'Action finished in {min} min {sec} sec')

    def closeWorkspace(self, objectName: str, isDelete: bool):
        self.switchTo(self.homeInterface)
        self.navigationInterface.removeWidget(objectName)
        self.homeInterface.updateRecentFile(objectName, isDelete)

    def openWorkspace(self, path: str):
        fileNameWithSuffix: str = os.path.basename(path)
        fileNameOnly: str = fileNameWithSuffix[0:fileNameWithSuffix.rindex(
            '.')]
        self.workInterface = WorkInterface(fileNameOnly, f'{fileNameOnly}.{FORMATTED_IMG_SUFFIX}',
                                           self.imageColorizer, self.simpleLama, self.maskGenerator, self.yolov7Detector, self)
        self.workInterface.setSpinnerEnabledSignal.connect(
            self.onSpinnerEnabledSet)
        self.addSubInterface(self.workInterface, FIF.EDIT, fileNameOnly)
        self.switchTo(self.workInterface)
        self.workInterface.setImage(path)
        self.workInterface.closeWorkspaceSignal.connect(self.closeWorkspace)

    def initImageColorizer(self):
        '''
        初始化图片色彩化器
        :return:
        '''
        # device.set(device=DeviceId.GPU0)
        # torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark = False
        warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
        self.imageColorizer = get_image_colorizer(
            root_folder=Path('./resource/'), artistic=False)

    def initSimpleLama(self):
        '''
        初始化 SimpleLama 物体移除器
        :return:
        '''
        print('Initializing SimpleLama')
        self.simpleLama = SimpleLama()
        print('SimpleLama initialized')

    def initYOLOv7(self):
        '''
        初始化 YOLOv7 检测器
        :return:
        '''
        print('Initializing YOLOv7')
        self.yolov7Detector = Yolov7Detector(traced=False)
        print('YOLOv7 initialized')

    def initSegmentPredictor(self):
        print('Initializing SegmentPredictor')
        sam = sam_model_registry["default"](
            checkpoint="./resource/models/sam_vit_h_4b8939.pth")
        # sam.to(device="cuda")
        sam.to(device="cpu")
        self.maskGenerator = SamPredictor(sam)
        print('SegmentPredictor initialized')

    # def initAutoMaskGenerator(self):
    #     if self.autoMaskGenerator is None:
    #         print('Initializing AutoMaskGenerator')
    #         sam = sam_model_registry["default"](
    #             checkpoint="./resource/models/sam_vit_h_4b8939.pth")
    #         # sam.to(device="cuda")
    #         sam.to(device="cpu")
    #         self.autoMaskGenerator = SamAutomaticMaskGenerator(sam)
    #         print('AutoMaskGenerator initialized')

    # def initVideoColorizer(self):
    #     '''
    #     初始化视频上色器
    #     :return:
    #     '''
    #     torch.backends.cudnn.benchmark = False
    #     self.videoColorizer = get_video_colorizer()
