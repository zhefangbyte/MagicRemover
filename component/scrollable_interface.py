from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QGridLayout, QSizePolicy
from qfluentwidgets import ScrollArea, TitleLabel, VBoxLayout, IndeterminateProgressBar

from common.style_sheet import StyleSheet


class ScrollableInterface(ScrollArea):

    def __init__(self, objectName: str, title: str, parent=None):
        super().__init__(parent=parent)

        StyleSheet.VIEW_INTERFACE.apply(self)

        self.pageWidget: QWidget = QWidget()
        self.pageWidget.setObjectName('pageWidget')

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setViewportMargins(0, 80, 0, 20)
        self.setWidget(self.pageWidget)
        self.setWidgetResizable(True)
        self.setObjectName(objectName)

        self.titleLabel: TitleLabel = TitleLabel(self.tr(title), self)
        self.titleLabel.move(36, 30)

        self.pageLayout: QGridLayout = QGridLayout(self.pageWidget)
        self.pageLayout.setContentsMargins(36, 10, 36, 0)
        self.pageLayout.setAlignment(Qt.AlignTop)
        self.pageLayout.setSpacing(14)
