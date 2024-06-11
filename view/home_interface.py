import os
import threading

import PIL.Image
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QWidget, QGridLayout
from qfluentwidgets import BodyLabel, SubtitleLabel, ImageLabel, FlowLayout, PrimaryPushButton, LineEdit, InfoBar, \
    CardWidget
from deoldify.visualize import *

from common.config import cfg, FORMATTED_IMG_SUFFIX
from common.util import Util
from component.scrollable_interface import ScrollableInterface

class HomeInterface(ScrollableInterface):
    """ Home interface """

    openWorkspaceSignal: pyqtSignal = pyqtSignal(str)

    def __init__(self, objectName: str, title: str, parent):
        super().__init__(objectName, title, parent)

        self.flowLayout: FlowLayout = FlowLayout()
        self.flowLayout.setContentsMargins(0, 0, 0, 0)

        self.filesWidget: QWidget = QWidget()
        self.filesWidget.setLayout(self.flowLayout)

        self.contLabel: SubtitleLabel = SubtitleLabel(text=self.tr('从最近编辑的文件继续'))

        itemList: list[str] = os.listdir(cfg.workFolder.value)

        self.noLastLabel: BodyLabel = BodyLabel(text=self.tr('没有最近编辑的文件'))
        self.noLastLabel.setVisible(itemList.__len__() == 0)

        for item in itemList:
            self.addFileToRecent(f'{cfg.workFolder.value}/{item}')

        newLocalFileLabel: SubtitleLabel = SubtitleLabel(text=self.tr('从本地文件创建'))

        chooseLocalFileBtn: PrimaryPushButton = PrimaryPushButton()
        chooseLocalFileBtn.setText(self.tr('选择文件'))
        chooseLocalFileBtn.clicked.connect(self.onChooseLocalFileBtnClicked)

        newRemoteFileLabel: SubtitleLabel = SubtitleLabel(text=self.tr('从网络资源创建'))

        self.newRemoteFileLineEdit: LineEdit = LineEdit()
        self.newRemoteFileLineEdit.setPlaceholderText(self.tr('http(s)://network.path.to/image_or_video'))

        self.confirmRemoteFileBtn: PrimaryPushButton = PrimaryPushButton()
        self.confirmRemoteFileBtn.setText(self.tr('确定'))
        self.confirmRemoteFileBtn.clicked.connect(self.onConfirmRemoteFileBtnClicked)

        self.checkUrlValidation()

        self.pageLayout.addWidget(self.contLabel, 0, 0, 1, 2)
        self.pageLayout.addWidget(self.noLastLabel, 1, 0, 1, 2)
        self.pageLayout.addWidget(self.filesWidget, 2, 0, 1, 2)
        self.pageLayout.addWidget(newLocalFileLabel, 3, 0, 1, 2)
        self.pageLayout.addWidget(chooseLocalFileBtn, 4, 0, 1, 2, alignment=Qt.AlignLeft)
        self.pageLayout.addWidget(newRemoteFileLabel, 5, 0, 1, 2)
        self.pageLayout.addWidget(self.newRemoteFileLineEdit, 6, 0, 1, 1)
        self.pageLayout.addWidget(self.confirmRemoteFileBtn, 6, 1, 1, 1, alignment=Qt.AlignLeft)

    def addFileToRecent(self, itemPath: str):
        item: ImageLabel = ImageLabel()
        image: PIL.Image.Image = PIL.Image.open(itemPath)
        qImage: QImage = image.toqimage()
        thumbnailHeight: int = 120
        thumbnailWidth: int = int(qImage.width() / qImage.height() * thumbnailHeight)
        image.thumbnail((thumbnailWidth, thumbnailHeight), PIL.Image.LANCZOS)
        item.setImage(image.toqimage())
        item.setBorderRadius(5, 5, 5, 5)

        itemLayout: QGridLayout = QGridLayout()
        itemLayout.setContentsMargins(3, 3, 3, 3)
        itemLayout.addWidget(item, 0, 0, 1, 1)

        itemWidget: CardWidget = CardWidget(self.filesWidget)
        itemWidget.setContentsMargins(0, 0, 0, 0)
        itemWidget.setToolTip(itemPath)
        itemWidget.setLayout(itemLayout)
        itemWidget.setFixedSize(thumbnailWidth + 6, thumbnailHeight + 6)
        itemWidget.clicked.connect(self.onImageClicked)

        self.flowLayout.addWidget(itemWidget)

    def updateRecentFile(self, fileName: str, isDelete: bool):
        print(f'Updating recent files: {fileName}')
        found: bool = False
        for child in self.filesWidget.children():
            if isinstance(child, CardWidget):
                if child.toolTip().endswith(f'{fileName}.{FORMATTED_IMG_SUFFIX}'):
                    if isDelete:
                        print(f'Removing {child.toolTip()} from UI')
                        self.flowLayout.removeWidget(child)
                        child.deleteLater()
                        return
                    else:
                        found = True
                        break
        if not found:
            print(f'Adding {fileName} to UI')
            self.addFileToRecent(f'{cfg.workFolder.value}/{fileName}.{FORMATTED_IMG_SUFFIX}')

    def checkUrlValidation(self):
        url: str = self.newRemoteFileLineEdit.text()
        return url.startswith('http://') or url.startswith('https://')

    def onChooseLocalFileBtnClicked(self):
        filePath: str = Util.getOpenFileName(self)

        if filePath == "":
            return

        self.openWorkspaceSignal.emit(filePath)

    def onConfirmRemoteFileBtnClicked(self):
        if self.checkUrlValidation():
            infoBar: InfoBar = InfoBar.info('提示', "正在下载图片", duration=-1, parent=self)

            def task():
                downloadedFilePath = Util.downloadImage(self.newRemoteFileLineEdit.text())
                infoBar.close()
                self.openWorkspaceSignal.emit(downloadedFilePath)

            threading.Thread(target=task).start()
        else:
            InfoBar.error('错误', "非法的链接格式", parent=self)

    def onImageClicked(self):
        self.openWorkspaceSignal.emit(self.sender().toolTip())
