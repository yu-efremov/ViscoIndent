# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 20:09:03 2020
work, checked, simple version with def
@author: Yuri
import_opt = 'single' - single file,
import_opt = 'multi' - multiple files
import_opt = 'folder' - foldername
import_opt = 'multi_from_folders' - multiple from several folders
import_opt = 'multi_folders' - many folder names
import_opt = 'get_subfolders' - subfolders of selected folder
import_opt = 'all_in_folder' - folder content
import_opt = 'all_in_folder_2level' - content of subfolders
import_opt == 'all_in_selected_folders' - select several folders (content)

"""

import sys, os, glob
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, \
                            QListView, QAbstractItemView, QTreeView


def file_import_dialog_qt5(import_opt='multi', file_types='*.*',
                           window_title='select files',
                           start_folder=''):
    print('start file selection')
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
            print('done file selection')
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
            print('done file selection')
            return filenames

    if import_opt == 'multi_from_folders':
        w.fileNames, _ = QFileDialog.getOpenFileNames(w, window_title, start_folder, file_types, options=options)
        filenames = []
        ij = 2
        while w.fileNames:
            print(w.fileNames)
            w.close()
            filenamesH = w.fileNames
            for filename in filenamesH:
                filenames.append(filename)
                window_title2 = window_title + ' from folder_' + str(ij) + ' or cancel'
            ij = ij + 1
            w.fileNames, _ = QFileDialog.getOpenFileNames(w, window_title2, start_folder, file_types, options=options)
        del w
        del app
        print('done file selection')
        return filenames

    if import_opt == 'folder':
        w.dir_path = QFileDialog.getExistingDirectory(w, "Choose Directory", start_folder)
        if w.dir_path:
            print(w.dir_path)
            w.close()
            foldername = w.dir_path
            del w
            del app
            print('done file selection')
            return foldername

    if import_opt == 'multi_folders':
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.DirectoryOnly)
        file_dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        file_dialog.setDirectory(start_folder)
        file_view = file_dialog.findChild(QListView, 'listView')

        # to make it possible to select multiple directories:
        if file_view:
            file_view.setSelectionMode(QAbstractItemView.MultiSelection)
        f_tree_view = file_dialog.findChild(QTreeView)
        if f_tree_view:
            f_tree_view.setSelectionMode(QAbstractItemView.MultiSelection)

        if file_dialog.exec():
            paths = file_dialog.selectedFiles()
        return paths

    if import_opt == 'get_subfolders':
        w.dir_path = QFileDialog.getExistingDirectory(w, "Choose Directory", start_folder)
        if w.dir_path:
            print(w.dir_path)
            w.close()
            foldername = w.dir_path
            paths = [ f.path for f in os.scandir(foldername) if f.is_dir() ]
            del w
            del app
            print('done file selection')
            return paths

    if import_opt == 'all_in_folder':
        w.dir_path = QFileDialog.getExistingDirectory(w, "Choose Directory", start_folder)
        if w.dir_path:
            print(w.dir_path)
            w.close()
            foldername = w.dir_path
            filenames = []
            # folderwithtype = foldername+"/"+file_types
            # print(glob.glob(folderwithtype))
            for file in glob.glob(foldername+"/"+file_types):
                filenames.append(file)
            del w
            del app
            print('done file selection')
            return filenames

    if import_opt == 'all_in_folder_2level':
        w.dir_path = QFileDialog.getExistingDirectory(w, "Choose Directory", start_folder)
        if w.dir_path:
            print(w.dir_path)
            w.close()
            foldername = w.dir_path
            filenames = []
            for file in glob.glob(foldername+"/*/"+file_types):
                filenames.append(file)
            del w
            del app
            print('done file selection')
            return filenames

    if import_opt == 'all_in_selected_folders':
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.DirectoryOnly)
        file_dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        file_dialog.setDirectory(start_folder)
        file_view = file_dialog.findChild(QListView, 'listView')

        # to make it possible to select multiple directories:
        if file_view:
            file_view.setSelectionMode(QAbstractItemView.MultiSelection)
        f_tree_view = file_dialog.findChild(QTreeView)
        if f_tree_view:
            f_tree_view.setSelectionMode(QAbstractItemView.MultiSelection)

        if file_dialog.exec():
            paths = file_dialog.selectedFiles()
        filenames = []
        for pathname in paths:
            files = glob.iglob(pathname+"/"+file_types)
            for file in files:
                filenames.append(file)
        del w
        del app
        print('done file selection')
        return filenames

if __name__ == '__main__':  #
    # multi_folders single multi multi_folders all_in_folder_2level all_in_selected_folders get_subfolders
    import_opt = 'multi_from_folders'  # 
    # import_opt = 'get_subfolders'
    # import_opt = 'all_in_selected_folders'
    file_types = '*.dat'  # '*.dat' '*.*'
    start_folder = 'D:/MailCloud/BioMomentum'
    start_folder = 'D:/MEGAsync/My materials/python/Ting_code/examples'
    # start_folder = 'D:/MailCloud/AFM_data/BrukerResolve'
    filename = file_import_dialog_qt5(import_opt=import_opt, start_folder=start_folder, file_types=file_types)  #
    # print(filename, 'selected')
