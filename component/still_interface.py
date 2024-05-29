from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QGridLayout, QSizePolicy
from qfluentwidgets import TitleLabel, BodyLabel

from common.style_sheet import StyleSheet


class StillInterface(QWidget):

    def __init__(self, objectName: str, title: str, parent=None):
        super().__init__(parent=parent)

        self.title = title

        StyleSheet.VIEW_INTERFACE.apply(self)

        self.pageWidget: QWidget = QWidget()
        self.pageWidget.setObjectName('pageWidget')

        self.setObjectName(objectName)

        self.titleLabel: TitleLabel = TitleLabel(text=self.tr(self.title))

        self.pageLayout: QGridLayout = QGridLayout(self)
        self.pageLayout.setAlignment(Qt.AlignTop)
        self.pageLayout.setSpacing(8)
        self.pageLayout.setContentsMargins(36, 30, 36, 20)
