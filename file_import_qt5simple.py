# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 20:09:03 2020
work, checked, simple version with def
@author: Yuri
import_opt = 'single' - single file,
import_opt = 'multi' - multiple files
import_opt = 'folder' - folder content
"""

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog


def file_import_dialog_qt5(import_opt='multi', file_types='*.*',
                           window_title='select files',
                           start_folder=''):
    print('start')
    # file_types = "All Files (*);;Python Files (*.py)"
    app = QApplication(sys.argv)
    w = QWidget()
    w.resize(250, 150)
    w.move(350, 350)
    w.setWindowTitle('Simple')
    w.show()
    options = QFileDialog.Options()
    if import_opt == 'single':
        w.fileName, _ = QFileDialog.getOpenFileName(w, window_title, start_folder, file_types, options=options)
        if w.fileName:
            print(w.fileName)
            w.close()
            filename = w.fileName
            del w
            del app
            print('done')
            return filename
            # sys.exit(app.exec_())  #freeze the system
    if import_opt == 'multi':
        w.fileNames, _ = QFileDialog.getOpenFileNames(w, window_title, start_folder, file_types, options=options)
        if w.fileNames:
            print(w.fileNames)
            w.close()
            filenames = w.fileNames
            del w
            del app
            print('done')
            return filenames
    if import_opt == 'folder':
        w.dir_path = QFileDialog.getExistingDirectory(w, "Choose Directory", start_folder)
        if w.dir_path:
            print(w.dir_path)
            w.close()
            filename = w.dir_path
            del w
            del app
            print('done')
            return filename


if __name__ == '__main__':  #
    import_opt = 'multi'  # single, 2 - multi
    file_types = '*.dat'
    start_folder = 'D:/MailCloud/BioMomentum'
    filename = file_import_dialog_qt5(import_opt=import_opt, start_folder=start_folder)  #
    # print(filename, 'selected')
