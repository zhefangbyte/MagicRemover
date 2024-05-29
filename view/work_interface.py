from typing import Any

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, pyqtSignal, QPointF
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtWidgets import QWidget, QHBoxLayout
from qfluentwidgets import FluentIcon as FIF, Action, TransparentDropDownPushButton, RoundMenu, \
    TransparentToolButton, TransparentToggleToolButton, SimpleCardWidget, \
    InfoBar, BodyLabel, ColorPickerButton, Flyout, FlyoutView, PushButton, SpinBox, \
    StrongBodyLabel, Icon, ImageLabel, LineEdit

from common.command import Command, CommandMode
from common.icon import FluentIconExt, CursorIcon
from common.quick_action import QuickAction, QuickActionMode
from common.util import Util
from component.draggable_image import DraggableImage
from component.still_interface import StillInterface


class WorkInterface(StillInterface):
    closeWorkspaceSignal: pyqtSignal = pyqtSignal(str, bool)
    setSpinnerEnabledSignal: pyqtSignal = pyqtSignal(bool)

    def __init__(self, objectName: str, title: str, parent=None):
        super().__init__(objectName, title, parent)

        self.imageWidget: DraggableImage = DraggableImage()
        self.imageWidget.setZoomFactorLabelTextSignal.connect(self.setZoomFactorLabelText)
        self.imageWidget.setUndoBtnEnabledSignal.connect(self.setUndoBtnEnabled)
        self.imageWidget.setRedoBtnEnabledSignal.connect(self.setRedoBtnEnabled)
        self.imageWidget.setSpinnerEnabledSignal.connect(self.setSpinnerEnabledSignal.emit)
        self.imageWidget.setColorLabelTextSignal.connect(self.setColorLabelText)

        self.bottomLeftCmdBar: QWidget = self.createBottomLeftCmdBar()

        self.pageLayout.addWidget(self.createTopLeftCmdBar(), 0, 0, 1, 1, alignment=Qt.AlignLeft)
        self.pageLayout.addWidget(self.createTopMiddleCmdBar(), 0, 1, 1, 3, alignment=Qt.AlignHCenter)
        self.pageLayout.addWidget(self.createTopRightCmdBar(), 0, 4, 1, 1, alignment=Qt.AlignRight)
        self.pageLayout.addWidget(self.imageWidget, 1, 0, 1, 5)
        self.pageLayout.addWidget(self.bottomLeftCmdBar, 2, 0, 1, 4, alignment=Qt.AlignLeft)
        self.pageLayout.addWidget(self.createBottomRightCmdBar(), 2, 4, 1, 1, alignment=Qt.AlignRight)

    def createTopRightCmdBar(self) -> QWidget:
        configCommand: TransparentToolButton = TransparentToolButton()

        def onConfigCmdClicked():
            view: FlyoutView = FlyoutView(
                title="配置",
                isClosable=True,
                content=""
            )

            def onColorChanged(color: QColor):
                self.imageWidget.penColor = color
                configCommand.click()

            penColor: QColor = self.imageWidget.penColor
            penColorLabel: StrongBodyLabel = StrongBodyLabel(text='当前画笔颜色')
            hiddenBtn: ColorPickerButton = ColorPickerButton(
                color=penColor,
                title="",
                parent=self
            )
            hiddenBtn.colorChanged.connect(onColorChanged)
            colorPickerBtn: PushButton = PushButton()
            colorPickerBtn.setText('更改')
            colorPickerBtn.clicked.connect(lambda: hiddenBtn.click())
            colorWidget: QWidget = self.createColorWidget(penColor)

            def penWidthFactorSpinBoxValueChanged(value: int):
                self.imageWidget.penWidthFactor = value

            penWidthFactorLabel: StrongBodyLabel = StrongBodyLabel(text='画笔粗细')
            penWidthFactorSpinBox: SpinBox = SpinBox()
            penWidthFactorSpinBox.setValue(self.imageWidget.penWidthFactor)
            penWidthFactorSpinBox.valueChanged.connect(penWidthFactorSpinBoxValueChanged)

            view.widgetLayout.addSpacing(10)
            view.addWidget(penColorLabel)
            view.addWidget(colorWidget)
            view.addWidget(colorPickerBtn)
            view.widgetLayout.addSpacing(10)
            view.addWidget(penWidthFactorLabel)
            view.addWidget(penWidthFactorSpinBox)
            view.widgetLayout.addSpacing(10)

            # show view
            w = Flyout.make(view, configCommand, self)
            view.closed.connect(w.close)

        configCommand.setIcon(FIF.SETTING)
        configCommand.clicked.connect(onConfigCmdClicked)
        Util.setModernTooltipText(configCommand, '配置')

        saveCommand: TransparentToolButton = TransparentToolButton()
        saveCommand.setIcon(FIF.SAVE)
        saveCommand.clicked.connect(self.onSaveCommandExecuted)
        saveCommand.setShortcut('Ctrl+S')
        Util.setModernTooltipText(saveCommand, '保存 Ctrl+S')

        exportCommand: TransparentToolButton = TransparentToolButton()
        exportCommand.setIcon(FluentIconExt.DUAL_SCREEN_MIRROR)
        exportCommand.clicked.connect(self.onExportCommandExecuted)
        exportCommand.setShortcut('Ctrl+Shift+E')
        Util.setModernTooltipText(exportCommand, '保存对比图片')

        closeCommand: TransparentToolButton = TransparentToolButton()
        closeCommand.setIcon(FIF.CLOSE)
        closeCommand.clicked.connect(self.onCloseCommandExecuted)
        closeCommand.setShortcut('Ctrl+W')
        Util.setModernTooltipText(closeCommand, '关闭 Ctrl+W')

        deleteCommand: TransparentToolButton = TransparentToolButton()
        deleteCommand.setIcon(FIF.DELETE)
        deleteCommand.clicked.connect(self.onDeleteCommandExecuted)
        Util.setModernTooltipText(deleteCommand, '删除 Del')

        layout: QHBoxLayout = QHBoxLayout()
        layout.setSpacing(0)
        layout.addWidget(configCommand)
        layout.addWidget(saveCommand)
        layout.addWidget(exportCommand)
        layout.addWidget(deleteCommand)
        layout.addWidget(closeCommand)

        widget: QWidget = QWidget()
        widget.setLayout(layout)

        return widget

    def createTopLeftCmdBar(self) -> QWidget:
        widget: SimpleCardWidget = SimpleCardWidget()

        self.onCommandClicked(Command(CommandMode.UNKNOWN), widget)

        pointSelectCommandLabel: str = '点选消除'
        pointSelectCommand: TransparentToggleToolButton = TransparentToggleToolButton(widget)
        pointSelectCommand.setIcon(FluentIconExt.GRID_DOTS)
        pointSelectCommand.clicked.connect(lambda: self.onCommandClicked(Command(CommandMode.POINT_SELECT_REMOVE, True), widget))
        Util.setModernTooltipText(pointSelectCommand, pointSelectCommandLabel)

        rectSelectCommandLabel: str = '框选消除'
        rectSelectCommand: TransparentToggleToolButton = TransparentToggleToolButton(widget)
        rectSelectCommand.setIcon(FluentIconExt.SELECT_OBJECT)
        rectSelectCommand.clicked.connect(lambda: self.onCommandClicked(Command(CommandMode.RECTANGLE_SELECT_REMOVE, True), widget))
        Util.setModernTooltipText(rectSelectCommand, rectSelectCommandLabel)

        paintSelectCommandLabel: str = '涂抹消除'
        paintSelectCommand: TransparentToggleToolButton = TransparentToggleToolButton(widget)
        paintSelectCommand.setIcon(FluentIconExt.HAND_DRAW)
        paintSelectCommand.clicked.connect(lambda: self.onCommandClicked(Command(CommandMode.PAINT_SELECT_REMOVE, True), widget))
        Util.setModernTooltipText(paintSelectCommand, paintSelectCommandLabel)

        dropperCommandLabel: str = '吸管'
        dropperCommand: TransparentToggleToolButton = TransparentToggleToolButton(widget)
        dropperCommand.setIcon(FluentIconExt.EYEDROPPER)
        dropperCommand.clicked.connect(lambda: self.onCommandClicked(Command(CommandMode.DROPPER), widget))
        Util.setModernTooltipText(dropperCommand, dropperCommandLabel)

        paintCommandLabel: str = '画笔'
        paintCommand: TransparentToggleToolButton = TransparentToggleToolButton(widget)
        paintCommand.setIcon(FluentIconExt.INKING_TOOL)
        paintCommand.clicked.connect(lambda: self.onCommandClicked(Command(CommandMode.DRAW), widget))
        Util.setModernTooltipText(paintCommand, paintCommandLabel)

        fillColorCommandLabel: str = '油漆桶'
        fillColorCommand: TransparentToggleToolButton = TransparentToggleToolButton(widget)
        fillColorCommand.setIcon(FluentIconExt.PAINT_BUCKET)
        fillColorCommand.clicked.connect(lambda: self.onCommandClicked(Command(CommandMode.PAINT_BUCKET, True), widget))
        fillColorCommand.setContextMenuPolicy(Qt.CustomContextMenu)
        Util.setModernTooltipText(fillColorCommand, fillColorCommandLabel)

        cropCommandLabel: str = '裁剪'
        cropCommand: TransparentToggleToolButton = TransparentToggleToolButton(widget)
        cropCommand.setIcon(FluentIconExt.CROP)
        cropCommand.clicked.connect(lambda: self.onCommandClicked(Command(CommandMode.CROP), widget))
        Util.setModernTooltipText(cropCommand, cropCommandLabel)

        layout: QHBoxLayout = QHBoxLayout()
        layout.setSpacing(0)
        layout.addWidget(pointSelectCommand)
        layout.addWidget(rectSelectCommand)
        layout.addWidget(paintSelectCommand)
        layout.addWidget(dropperCommand)
        layout.addWidget(paintCommand)
        layout.addWidget(fillColorCommand)
        layout.addWidget(cropCommand)

        widget.setLayout(layout)

        return widget

    def createTopMiddleCmdBar(self) -> QWidget:
        imgOverlayCommand: TransparentToolButton = TransparentToolButton()
        imgOverlayCommand.setIcon(FluentIconExt.IMAGE_SHADOW)
        imgOverlayCommand.clicked.connect(lambda: self.onQuickActionClicked(QuickAction(QuickActionMode.IMAGE_OVERLAY),
                                                                            Util.getOpenFileName(self)))
        Util.setModernTooltipText(imgOverlayCommand, '叠加底片')

        colorOverlayCommand: TransparentToolButton = TransparentToolButton()
        colorOverlayCommand.setIcon(FluentIconExt.COLOR_BACKGROUND)
        colorOverlayCommand.clicked.connect(lambda: self.onQuickActionClicked(QuickAction(QuickActionMode.COLOR_OVERLAY)))
        Util.setModernTooltipText(colorOverlayCommand, '叠加底色')

        removeBgCommand: TransparentToolButton = TransparentToolButton()
        removeBgCommand.setIcon(FluentIconExt.VIDEO_BACKGROUND_EFFECT)
        removeBgCommand.clicked.connect(lambda: self.onQuickActionClicked(QuickAction(QuickActionMode.REMOVE_BACKGROUND, True)))
        Util.setModernTooltipText(removeBgCommand, '移除背景')

        purifyBgCommand: TransparentToolButton = TransparentToolButton()
        purifyBgCommand.setIcon(FluentIconExt.IMAGE_ARROW_COUNTERCLOCKWISE)
        purifyBgCommand.clicked.connect(lambda: self.onQuickActionClicked(QuickAction(QuickActionMode.PURIFY_BACKGROUND, True)))
        Util.setModernTooltipText(purifyBgCommand, '整体去杂')

        colorizeCommand: TransparentToolButton = TransparentToolButton()
        colorizeCommand.setIcon(FIF.BRUSH)
        colorizeCommand.clicked.connect(lambda: self.onQuickActionClicked(QuickAction(QuickActionMode.BRUSH, True)))
        Util.setModernTooltipText(colorizeCommand, '上色')

        rotateCommand: TransparentToolButton = TransparentToolButton()
        rotateCommand.setIcon(FIF.ROTATE)
        rotateCommand.clicked.connect(lambda: self.onQuickActionClicked(QuickAction(QuickActionMode.ROTATE)))
        rotateCommand.setShortcut('Ctrl+R')
        Util.setModernTooltipText(rotateCommand, '旋转 Ctrl+R')

        self.undoCommand: TransparentToolButton = TransparentToolButton()
        self.undoCommand.setIcon(FluentIconExt.ARROW_UNDO)
        self.undoCommand.clicked.connect(lambda: self.onQuickActionClicked(QuickAction(QuickActionMode.UNDO)))
        self.undoCommand.setShortcut('Ctrl+Z')
        Util.setModernTooltipText(self.undoCommand, '撤销 Ctrl+Z')

        self.redoCommand: TransparentToolButton = TransparentToolButton()
        self.redoCommand.setIcon(FluentIconExt.ARROW_REDO)
        self.redoCommand.clicked.connect(lambda: self.onQuickActionClicked(QuickAction(QuickActionMode.REDO)))
        self.redoCommand.setShortcut('Ctrl+Y')
        Util.setModernTooltipText(self.redoCommand, '重做 Ctrl+Y')

        layout: QHBoxLayout = QHBoxLayout()
        layout.setSpacing(0)
        layout.addWidget(colorOverlayCommand)
        layout.addWidget(imgOverlayCommand)
        layout.addWidget(removeBgCommand)
        layout.addWidget(purifyBgCommand)
        layout.addWidget(colorizeCommand)
        layout.addWidget(rotateCommand)
        layout.addWidget(self.undoCommand)
        layout.addWidget(self.redoCommand)

        widget: SimpleCardWidget = SimpleCardWidget()
        widget.setLayout(layout)

        return widget

    def createBottomLeftCmdBar(self) -> QWidget:

        lineEdit: QWidget = LineEdit()

        submitBtn: TransparentToolButton = TransparentToolButton(FIF.ACCEPT)
        submitBtn.clicked.connect(lambda: self.onQuickActionClicked(QuickAction(QuickActionMode.TEXT, True), lineEdit.text()))

        layout: QHBoxLayout = QHBoxLayout()
        layout.setSpacing(5)
        layout.addWidget(lineEdit)
        layout.addWidget(submitBtn)

        widget: SimpleCardWidget = SimpleCardWidget()
        widget.setLayout(layout)

        return widget

    def createBottomRightCmdBar(self) -> QWidget:

        zoomOutCommand: TransparentToolButton = TransparentToolButton()
        zoomOutCommand.setIcon(FIF.REMOVE)
        zoomOutCommand.setShortcut('Ctrl+-')
        zoomOutCommand.clicked.connect(lambda: self.onQuickActionClicked(QuickAction(QuickActionMode.ZOOM_OUT)))
        Util.setModernTooltipText(zoomOutCommand, '缩小 Ctrl+-')

        scaleToWindowAction: Action = Action(
            FIF.FIT_PAGE,
            '适应窗口大小',
            triggered=lambda: self.onQuickActionClicked(QuickAction(QuickActionMode.SCALE_TO_WINDOW)))
        scaleToWindowAction.setShortcut('Ctrl+0')

        scaleToRawAction: Action = Action(
            FluentIconExt.RESIZE_LARGE,
            '显示原尺寸',
            triggered=lambda: self.onQuickActionClicked(QuickAction(QuickActionMode.SCALE_TO_RAW)))
        scaleToRawAction.setShortcut('Ctrl+1')

        zoomFactorBtnMenu: RoundMenu = RoundMenu()
        zoomFactorBtnMenu.addAction(scaleToWindowAction)
        zoomFactorBtnMenu.addAction(scaleToRawAction)

        self.zoomFactorBtn: TransparentDropDownPushButton = TransparentDropDownPushButton()
        self.zoomFactorBtn.setMenu(zoomFactorBtnMenu)
        Util.setModernTooltipText(self.zoomFactorBtn, '缩放')

        self.setZoomFactorLabelText()

        zoomInCommand: TransparentToolButton = TransparentToolButton()
        zoomInCommand.setIcon(FIF.ADD)
        zoomInCommand.setShortcut('Ctrl+=')
        zoomInCommand.clicked.connect(lambda: self.onQuickActionClicked(QuickAction(QuickActionMode.ZOOM_IN)))
        Util.setModernTooltipText(zoomInCommand, '放大 Ctrl+=')

        layout: QHBoxLayout = QHBoxLayout()
        layout.setSpacing(0)
        layout.addWidget(zoomOutCommand)
        layout.addWidget(self.zoomFactorBtn)
        layout.addWidget(zoomInCommand)

        widget: SimpleCardWidget = SimpleCardWidget()
        widget.setLayout(layout)

        return widget

    def createColorWidget(self, color: QColor) -> QWidget:
        textColor: QColor = \
            QColor(Qt.white) if color.lightness() <= 127 else QColor(Qt.black)
        self.colorNameLabel: BodyLabel = BodyLabel(text=f'{color.name().upper()}')
        self.colorNameLabel.setTextColor(textColor, textColor)

        colorLayout: QHBoxLayout = QHBoxLayout()
        colorLayout.setContentsMargins(5, 0, 5, 0)
        colorLayout.addWidget(self.colorNameLabel, alignment=Qt.AlignVCenter)
        self.colorWidget: QWidget = QWidget()
        self.colorWidget.setObjectName('colorWidget')
        self.colorWidget.setLayout(colorLayout)
        self.colorWidget.setStyleSheet(
            "#colorWidget {" +
            "background-color: " + color.name() + ";" +
            "border-radius: 5px;" +
            "}"
        )
        return self.colorWidget

    def setImage(self, path: str):
        self.imageWidget.initImage(path, self.title)

    def enterEvent(self, a0: QtCore.QEvent) -> None:
        self.imageWidget.setFocus()

    def onCommandClicked(self, command: Command, parent: QWidget):
        for child in parent.children():
            if isinstance(child, TransparentToggleToolButton):
                if child == self.sender():
                    self.imageWidget.onCommandClicked(command if child.isChecked() else CommandMode.UNKNOWN)
                else:
                    child.setChecked(False)
        if command.mode == CommandMode.UNKNOWN:
            self.imageWidget.onCommandClicked(command)

    def onQuickActionClicked(self, quickAction: QuickAction, arg: Any = None):
        self.imageWidget.onQuickActionBarClicked(quickAction, arg)

    def setZoomFactorLabelText(self):
        self.zoomFactorBtn.setText(self.tr("{:.0f}%".format(self.imageWidget.zoomFactor * 100)))

    def setUndoBtnEnabled(self, enabled: bool):
        self.undoCommand.setEnabled(enabled)

    def setRedoBtnEnabled(self, enabled: bool):
        self.redoCommand.setEnabled(enabled)

    def setColorLabelText(self):
        self.createColorWidget(self.imageWidget.penColor)

    def onCloseCommandExecuted(self):
        self.closeWorkspaceSignal.emit(self.objectName(), False)

    def onDeleteCommandExecuted(self):
        self.imageWidget.deleteStorageFile()
        self.closeWorkspaceSignal.emit(self.objectName(), True)

    def onSaveCommandExecuted(self):
        if self.imageWidget.saveCurrentSnapshotToWorkFolder():
            InfoBar.success('保存成功', "图片已保存至程序工作目录", parent=self)

    def onExportCommandExecuted(self):
        dirPath: str = Util.getExistingDirectory(self)
        if dirPath != '':
            self.imageWidget.saveComparedImages(dirPath)
            InfoBar.success('保存成功', f"对比图片已保存至{dirPath}", parent=self)
