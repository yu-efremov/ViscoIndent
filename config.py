# -*- coding: utf-8 -*-
"""
file with basic cellsuration settings


"""
from Pars_class import Pars_gen

config = {}
# start folder
config['start_folder'] = ''  # write data containing folder here
config['supress_ROIquestion'] = 0  # 0 - ask question, 1 - response no, 2 - yes
config['HeightfromZ'] = 0
config['ROImode'] = 0    # 0 - "no selection", 1 - "select above h-level",
                        # 2 - "draw", 3- "draw and select above h-level"
config['h_level'] = -10000  # nm, low value to select all points


cells = {}
# start folder
cells['start_folder'] = ''  # write data containing folder here
cells['supress_ROIquestion'] = 2  # 0 - ask question, 1 - response no, 2 - yes
cells['HeightfromZ'] = 1
cells['ROImode'] = 3    # 0 - "no selection", 1 - "select above h-level",
                        # 2 - "draw", 3- "draw and select above h-level"
cells['h_level'] = 200  # nm, select points above substarte level



