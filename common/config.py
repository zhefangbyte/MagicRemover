# coding:utf-8
from enum import Enum

from PyQt5.QtCore import QLocale
from qfluentwidgets import (qconfig, QConfig, ConfigItem, OptionsConfigItem, OptionsValidator, Theme, FolderValidator,
                            ConfigSerializer)


class Language(Enum):
    """ Language enumeration """

    CHINESE_SIMPLIFIED: QLocale = QLocale(QLocale.Chinese, QLocale.China)
    ENGLISH: QLocale = QLocale(QLocale.English)
    AUTO: QLocale = QLocale()


class LanguageSerializer(ConfigSerializer):
    """ Language serializer """

    def serialize(self, language) -> str:
        return language.value.name() if language != Language.AUTO else "Auto"

    def deserialize(self, value: str) -> Language:
        return Language(QLocale(value)) if value != "Auto" else Language.AUTO


class Config(QConfig):
    """ Config of application """

    # folders
    workFolder: ConfigItem = ConfigItem(
        "Folders", "Workspace", "app/workspace", FolderValidator())

    cacheFolder: ConfigItem = ConfigItem(
        "Folders", "Cache", "app/cache", FolderValidator())

    # main window
    language: OptionsConfigItem = OptionsConfigItem(
        "MainWindow", "Language", Language.AUTO, OptionsValidator(Language), LanguageSerializer(), restart=True)


YEAR: int = 2024
AUTHOR: str = '方喆'
VERSION: str = '1.0.0'
REPO_URL: str = 'https://github.com/founchoo/MagicRemover'
FORMATTED_IMG_SUFFIX: str = 'png'


cfg: Config = Config()
cfg.themeMode.value = Theme.AUTO
qconfig.load('app/config/config.json', cfg)
