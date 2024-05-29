import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from window.main_window import MainWindow

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)

    w = MainWindow()
    w.show()
    app.exec_()
