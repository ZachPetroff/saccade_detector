import numpy as np
import pandas as pd
import math
import warnings

gazes = pd.read_csv('gaze_positions.csv')
pupils = pd.read_csv('pupil_positions.csv')

gazes['gaze_timestamp'] = np.array(gazes['gaze_timestamp']) - np.array(gazes['gaze_timestamp'])[0]
pupils['pupil_timestamp'] = np.array(pupils['pupil_timestamp']) - np.array(pupils['pupil_timestamp'])[0]

gazes = gazes[gazes['confidence'] >= 0.9]
pupils = pupils[pupils['confidence'] >= 0.9]
pupils = pupils.dropna(subset=['circle_3d_normal_x'])
pupils['angleX'] = np.arctan(pupils['circle_3d_normal_x']/pupils['circle_3d_normal_z']) * (180/math.pi)
pupils['angleY'] = np.arctan(pupils['circle_3d_normal_y']/pupils['circle_3d_normal_z']) * (180/math.pi)

pupilR = pupils[pupils['eye_id'] == 0]
pupilL = pupils[pupils['eye_id'] == 1]

pupilR['angleX'] = np.array(pupilR['angleX']) * -1
pupilR['angleY'] = np.array(pupilR['angleY']) * -1





