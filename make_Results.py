# -*- coding: utf-8 -*-
"""
make pandas Results array for Tingsprocessing code
"""

import numpy as np
import pandas as pd


def make_Results(testlen):
    # testlen = 4096
    Results = pd.DataFrame(columns=['Name', 'Pixel', 'model', 'cpHertz',
                                    'EHertz', 'hysteresis', 'AdjRsqHertz',
                                    'max_force', 'max_adh', 'ind_depth',
                                    'Height', 'cpTing', 'resnorm', 'S',
                                    'AdjRsq', 'contact_timeH', 'EHertzBEC',
                                    'E0', 'Einf', 'alpha_tau', 'contact_timeT',
                                    'Freq', 'Estor', 'Eloss', 'E0BEC',
                                    'EinfBEC', 'alpha_tauBEC',
                                    'comment'], index=range(testlen))
    Results = Results.astype({'Name': str,
                              'Pixel': object,
                              'model': str,
                              'cpHertz': np.float32,
                              'EHertz': np.float32,
                              'hysteresis': np.float32,
                              'AdjRsqHertz': np.float32,
                              'max_force': np.float32,
                              'max_adh': np.float32,
                              'ind_depth': np.float32,
                              'Height': np.float32,
                              'cpTing': np.float32,
                              'resnorm': np.float32,
                              'AdjRsq': np.float32,
                              'S': np.float32,
                              'contact_timeH': np.float32,
                              'EHertzBEC': np.float32,
                              'E0': np.float32,
                              'Einf': np.float32,
                              'alpha_tau': np.float32,
                              'contact_timeT': np.float32,
                              'Freq': np.float32,
                              'Estor': np.float32,
                              'Eloss': np.float32,
                              'E0BEC': np.float32,
                              'EinfBEC': np.float32,
                              'alpha_tauBEC': np.float32,
                              'comment': str})
    return Results


if __name__ == '__main__':  #
    Resultstest = make_Results(10)
