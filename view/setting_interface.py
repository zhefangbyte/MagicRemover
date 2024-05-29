# coding:utf-8
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QFileDialog
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import InfoBar
from qfluentwidgets import (SettingCardGroup, OptionsSettingCard, PushSettingCard,
                            PrimaryPushSettingCard, ComboBoxSettingCard, CustomColorSettingCard,
                            setTheme, setThemeColor)

from common.config import cfg, AUTHOR, VERSION, YEAR, REPO_URL
from component.scrollable_interface import ScrollableInterface


class SettingInterface(ScrollableInterface):
    """ Setting interface """

    def __init__(self, objectName: str, title: str, parent=None):
        super().__init__(objectName, title, parent)

        self.dataGroup = SettingCardGroup(
            self.tr("数据"))
        self.workspaceFolderCard = PushSettingCard(
            self.tr('选择文件夹'),
            FIF.FOLDER,
            self.tr("程序工作目录"),
            cfg.get(cfg.workFolder),
            self.dataGroup
        )
        self.cacheFolderCard = PushSettingCard(
            self.tr('选择文件夹'),
            FIF.FOLDER,
            self.tr("程序缓存目录"),
            cfg.get(cfg.cacheFolder),
            self.dataGroup
        )

        self.personalGroup = SettingCardGroup(
            self.tr('个性化'))
        self.themeCard = OptionsSettingCard(
            cfg.themeMode,
            FIF.BRUSH,
            self.tr('主题明度'),
            self.tr("改变应用的明度"),
            texts=[
                self.tr('浅色'), self.tr('深色'),
                self.tr('跟随系统')
            ],
            parent=self.personalGroup
        )
        self.themeColorCard = CustomColorSettingCard(
            cfg.themeColor,
            FIF.PALETTE,
            self.tr('主题颜色'),
            self.tr('改变应用的主题色'),
            self.personalGroup
        )
        self.languageCard = ComboBoxSettingCard(
            cfg.language,
            FIF.LANGUAGE,
            self.tr('语言'),
            self.tr('设置语言'),
            texts=['简体中文', 'English', self.tr('跟随系统')],
            parent=self.personalGroup
        )

        self.aboutGroup = SettingCardGroup(self.tr('关于'))
        self.aboutCard = PrimaryPushSettingCard(
            self.tr('项目源码'),
            FIF.INFO,
            self.tr('关于'),
            '© ' + self.tr('Copyright') + f" {YEAR}, {AUTHOR}. " +
            self.tr('版本') + " " + VERSION,
            self.aboutGroup
        )

        self.dataGroup.addSettingCard(self.workspaceFolderCard)
        self.dataGroup.addSettingCard(self.cacheFolderCard)

        self.personalGroup.addSettingCard(self.themeCard)
        self.personalGroup.addSettingCard(self.themeColorCard)
        self.personalGroup.addSettingCard(self.languageCard)

        self.aboutGroup.addSettingCard(self.aboutCard)

        self.pageLayout.addWidget(self.dataGroup)
        self.pageLayout.addWidget(self.personalGroup)
        self.pageLayout.addWidget(self.aboutGroup)

        self.__connectSignalToSlot()

    def __showRestartTooltip(self):
        """ show restart tooltip """
        InfoBar.success(
            self.tr('Updated successfully'),
            self.tr('Configuration takes effect after restart'),
            duration=1500,
            parent=self
        )

    def __onWorkspaceFolderCardClicked(self):
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("选择文件夹"), "./")
        if not folder or cfg.get(cfg.workFolder) == folder:
            return

        cfg.set(cfg.workFolder, folder)
        self.workspaceFolderCard.setContent(folder)

    def __onCacheFolderCardClicked(self):
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("选择文件夹"), "./")
        if not folder or cfg.get(cfg.cacheFolder) == folder:
            return

        cfg.set(cfg.cacheFolder, folder)
        self.cacheFolderCard.setContent(folder)

    def __connectSignalToSlot(self):
        """ connect signal to slot """
        cfg.appRestartSig.connect(self.__showRestartTooltip)

        self.workspaceFolderCard.clicked.connect(self.__onWorkspaceFolderCardClicked)
        self.cacheFolderCard.clicked.connect(self.__onCacheFolderCardClicked)

        # personalization
        self.themeCard.optionChanged.connect(lambda ci: setTheme(cfg.get(ci)))
        self.themeColorCard.colorChanged.connect(setThemeColor)

        # about
        self.aboutCard.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(REPO_URL)))
