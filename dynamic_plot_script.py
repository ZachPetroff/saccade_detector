import tkinter as tk
import cv2
from PIL import ImageTk, Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import warnings
import sys

SUBJECT = sys.argv[1]
score_df = pd.read_csv(SUBJECT+'scores.csv',index_col=[0])
score_times = list(score_df['Time'])
scores = list(score_df['Score'])
remove_times = []

for s in range(len(scores)):
    if int(scores[s]) == 0:
        remove_times.append(round(float(score_times[s]),2))

def binsacc(sacl, sacr):
    NL = sacl.shape[0]  # number of microsaccades (left eye)
    NR = sacr.shape[0]  # number of microsaccades (right eye)
    sac = []
    
    for i in range(NL):  # loop over left-eye saccades
        l1 = sacl[i, 0]  # begin saccade left eye
        l2 = sacl[i, 1]  # end saccade left eye
        
        if NR > 0:
            R1 = sacr[:, 0]  # begin saccade right eye
            R2 = sacr[:, 1]  # end saccade right eye
            
            # testing for temporal overlap with right eye
            overlap = np.where((R2 >= l1) & (R1 <= l2))[0]
            
            if len(overlap) > 0:
                # define parameters for binocular saccades
                r1 = R1[overlap[0]]
                r2 = R2[overlap[0]]
                vl = sacl[i, 2]
                vr = sacr[overlap[0], 2]
                ampl = sacl[i, 3]
                ampr = sacr[overlap[0], 3]
                dxl = sacl[i, 5]
                dyl = sacl[i, 6]
                dxr = sacr[overlap[0], 5]
                dyr = sacr[overlap[0], 6]
                dx = dxl + dxr  
                dy = dyl + dyr
                phi = 180 / np.pi * np.arctan2(dy, dx)
                
                s = [min([l1, r1]), max([l2, r2]),
                     np.mean([vl, vr]), np.mean([ampl, ampr]),
                     phi, np.mean([dxl, dxr]), np.mean([dyl, dyr]),
                     r1, r2, l1, l2]
                
                # store all binocular saccades
                sac.append(s)
    sac = np.array(sac)
    
    # check if all saccades are separated by >= 3 samples
    nsac = sac.shape[0]
    k = 0
    
    while k < nsac-1:
        if sac[k, 1] + 3 <= sac[k + 1, 0]:
            k += 1
        else:
            sac[k, 1] = sac[k + 1, 1]
            sac[k, 2] = max([sac[k, 2], sac[k + 1, 2]])
            dx1 = sac[k, 5]
            dy1 = sac[k, 6]
            dx2 = sac[k + 1, 5]
            dy2 = sac[k + 1, 6]
            dx = dx1 + dx2
            dy = dy1 + dy2
            amp = np.sqrt(dx**2 + dy**2)
            phi = 180 / np.pi * np.arctan2(dy, dx)
            sac[k, 3] = amp
            sac[k, 4] = phi
            sac[k, 5] = dx
            sac[k, 6] = dy
            sac = np.delete(sac, k + 1, axis=0)
            nsac = nsac - 1
            
    return sac

def binsaccT(sacl,sacr):
    NL = sacl.shape[0]  # number of microsaccades (left eye)
    NR = sacr.shape[0]  # number of microsaccades (right eye)
    sac = []
    
    for i in range(NL):  # loop over left-eye saccades
        l1 = sacl[i, 0]  # begin saccade left eye
        l2 = sacl[i, 1]  # end saccade left eye
    
        l1t = sacl[i, 7]  # begin saccade left eye (time)
        l2t = sacl[i, 8]  # end saccade left eye (time)
    
        if NR > 0:
            R1 = sacr[:, 0]  # begin saccade right eye
            R2 = sacr[:, 1]  # end saccade right eye
    
            R1t = sacr[:, 7]  # begin saccade right eye (time)
            R2t = sacr[:, 8]  # end saccade right eye (time)
    
            # testing for temporal overlap with right eye
            overlap = np.where((R2t >= l1t) & (R1t <= l2t))[0]
    
            if len(overlap) > 0:
                # define parameters for binocular saccades
                r1 = R1[overlap[0]]
                r2 = R2[overlap[0]]
                vl = sacl[i, 2]
                vr = sacr[overlap[0], 2]
                ampl = sacl[i, 3]
                ampr = sacr[overlap[0], 3]
                dxl = sacl[i, 5]
                dyl = sacl[i, 6]
                dxr = sacr[overlap[0], 5]
                dyr = sacr[overlap[0], 6]
                dx = dxl + dxr
                dy = dyl + dyr
                phi = 180 / np.pi * np.arctan2(dy, dx)
    
                s = [min([l1, r1]), max([l2, r2]),
                     np.mean([vl, vr]), np.mean([ampl, ampr]),
                     phi, np.mean([dxl, dxr]), np.mean([dyl, dyr]),
                     r1, r2, l1, l2]
    
                # store all binocular saccades
                sac.append(s)
    sac = np.array(sac)
    return sac

def microsacc2_nlp2(x, vel, VFAC, MINDUR):
    DEBUG = 0
    sac = []
    radius = []

    # compute threshold
    msdx = np.sqrt(np.median(vel[:,0]**2) - np.median(vel[:,0])**2)
    msdy = np.sqrt(np.median(vel[:,1]**2) - np.median(vel[:,1])**2)
    
    if msdx < np.finfo(float).tiny:
        msdx = np.sqrt(np.mean(vel[:,0]**2) - np.mean(vel[:,0])**2)
        if msdx < np.finfo(float).tiny:
            warnings.warn('msdx < realmin in microsacc2_nlp2')
            return sac, radius
    
    if msdy < np.finfo(float).tiny:
        msdy = np.sqrt(np.mean(vel[:,1]**2) - np.mean(vel[:,1])**2)
        if msdy < np.finfo(float).tiny:
            warnings.warn('msdy < realmin in microsacc2_nlp2')
            return sac, radius
    
    radiusx = VFAC * msdx
    radiusy = VFAC * msdy
    radius = [radiusx, radiusy]
    
    # compute test criterion: ellipse equation
    test = (vel[:,0] / radiusx) ** 2 + (vel[:,1] / radiusy) ** 2
    indx = np.where(test > 1)[0]
    
    # determine saccades
    N = len(indx)-1
    tmp_sac = []
    nsac = 0
    dur = 1
    a = 0
    k = 0
    while k < N:
        if indx[k+1] - indx[k] == 1:
            dur += 1
        else:
            if dur >= MINDUR:
                nsac += 1
                b = k
                tmp_sac.append([indx[a], indx[b]])
            a = k+1
            dur = 1
        k += 1

    # check for minimum duration
    if dur >= MINDUR:
        nsac += 1
        b = k
        tmp_sac.append([indx[a], indx[b]])

    # At this point 'tmp_sac' has a list of starting and ending points for saccades
    # Now we need to check whether the eye was at rest before the
    # saccade. This is done by checking whether the mean x & y eye positions fit
    # within a predefined window

    KS = 20  # buffer window before and after a saccade
    WIN = 15  # window of noise that is allowed to count a saccade as "good"
    nsac2 = 0
    sac_good = []
    
    for ii in range(len(tmp_sac)):
        sac_good.append(0)
        
        if tmp_sac[ii][0] > 20:
            s1 = tmp_sac[ii][0] - KS
            s2 = tmp_sac[ii][1] + KS

            xm = np.mean(x[s1:s1+KS, 0])
            ym = np.mean(x[s1:s1+KS, 1])

            x_data = x[s1:s1+KS, 0]
            y_data = x[s1:s1+KS, 1]

            x_diff = x_data - xm > WIN
            y_diff = y_data - ym > WIN

            if np.max(x_diff) != 1 and np.max(y_diff) != 1:
                sac_good[ii] = 1

            if DEBUG:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 2)
                axes[0].plot(x[s1:s2, 0], 'r-')
                axes[0].plot(x[s1:s2, 1], 'b-')
                axes[0].plot([1, KS], [xm, xm], 'r--', linewidth=2)
                axes[0].plot([1, KS], [xm+WIN, xm+WIN], 'k--', linewidth=1)
                axes[0].plot([1, KS], [xm-WIN, xm-WIN], 'k:', linewidth=1)
                axes[0].plot([1, KS], [ym, ym], 'b--', linewidth=2)
                axes[0].plot([1, KS], [ym+WIN, ym+WIN], 'k--', linewidth=1)
                axes[0].plot([1, KS], [ym-WIN, ym-WIN], 'k:', linewidth=1)
                axes[0].set_title('Sac Good = ' + str(sac_good[ii]))

                axes[1].plot(vel[s1:s2, 0], 'r-')
                axes[1].plot(vel[s1:s2, 1], 'b-')
                plt.show()
        
        if sac_good[ii] == 1:
            nsac2 += 1
            sac.append(tmp_sac[ii])
    
    # compute peak velocity, horizontal and vertical components
    for s in range(nsac2):
        a = sac[s][0]
        b = sac[s][1]
        vpeak = np.max(np.sqrt(vel[a:b, 0]**2 + vel[a:b, 1]**2))
        dx = x[b, 0] - x[a, 0]
        dy = x[b, 1] - x[a, 1]
        
        i = np.arange(a, b+1)
        minx, ix1 = np.min(x[i, 0]), np.argmin(x[i, 0])
        maxx, ix2 = np.max(x[i, 0]), np.argmax(x[i, 0])
        miny, iy1 = np.min(x[i, 1]), np.argmin(x[i, 1])
        maxy, iy2 = np.max(x[i, 1]), np.argmax(x[i, 1])
        
        dX = np.sign(ix2 - ix1) * (maxx - minx)
        dY = np.sign(iy2 - iy1) * (maxy - miny)
        
        sac[s] = [a, b, vpeak, dx, dy, dX, dY]
    
    return sac, radius


def vecvel(xx, SAMPLING, TYPE): 
    N = xx.shape[0]  # length of the time series
    v = np.zeros(xx.shape)
    
    if TYPE == 1:
        v[1:N-2, :] = SAMPLING/2 * (xx[2:N-1, :] - xx[:N-3, :])
    elif TYPE == 2:
        v[2:N-3, :] = SAMPLING/6 * (xx[4:N-1, :] + xx[3:N-2, :] - xx[1:N-4, :] - xx[:N-5, :])
        v[1, :] = SAMPLING/2 * (xx[2, :] - xx[0, :])
        v[N-2, :] = SAMPLING/2 * (xx[-1, :] - xx[-3, :])
    elif TYPE == 3:
        if SAMPLING == 1000:
            n = 10
            Xm2 = (xx[n-9:-19, :] + xx[n-8:-18, :] + xx[n-7:-17, :] + xx[n-6:-16, :]) / 4
            Xm1 = (xx[n-5:-15, :] + xx[n-4:-14, :] + xx[n-3:-13, :] + xx[n-2:-12, :]) / 4
            Xp1 = (xx[n+5:-5, :] + xx[n+4:-6, :] + xx[n+3:-7, :] + xx[n+2:-8, :]) / 4
            Xp2 = (xx[n+9:, :] + xx[n+8:-1, :] + xx[n+7:-2, :] + xx[n+6:-3, :]) / 4
            v[n:N-(n-1), :] = (SAMPLING * (Xp2 + Xp1 - Xm1 - Xm2)) / 24
            v_strt = vecvel(xx[:14, :], 500, 3) * 2
            v_end = vecvel(xx[-13:, :], 500, 3) * 2
            v[:9, :] = v_strt[:9, :]
            v[-9:, :] = v_end[-9:, :]
        elif SAMPLING == 500:
            n = 5
            Xm2 = (xx[n-4:-9, :] + xx[n-3:-8, :]) / 2
            Xm1 = (xx[n-2:-7, :] + xx[n-1:-6, :]) / 2
            Xp1 = (xx[n+2:-3, :] + xx[n+1:-4, :]) / 2
            Xp2 = (xx[n+4:, :] + xx[n+3:-1, :]) / 2
            v[n:N-(n-1), :] = (SAMPLING * (Xp2 + Xp1 - Xm1 - Xm2)) / 12
            v[2:4, :] = SAMPLING/6 * (xx[4:6, :] + xx[3:5, :] - xx[1:3, :] - xx[:2, :])
            v[N-4:N-2, :] = SAMPLING/6 * (xx[N-2:, :] + xx[N-3:N-1, :] - xx[N-5:N-3, :] - xx[N-6:N-4, :])
            v[1, :] = SAMPLING/2 * (xx[2, :] - xx[0, :])
            v[N-2, :] = SAMPLING/2 * (xx[-1, :] - xx[-3, :])
            
    return v

gazes = pd.read_csv(SUBJECT+'gaze_positions.csv')
pupils = pd.read_csv(SUBJECT+'pupil_positions.csv')

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

pupilRcomb = np.array([ np.array(pupilR['angleX']), np.array(pupilR['angleY'])])
pupilLcomb = np.array([ np.array(pupilL['angleX']), np.array(pupilL['angleY'])])

x_gaze = np.array(gazes['norm_pos_x'])
y_gaze = np.array(gazes['norm_pos_y'])

x_bool1 = x_gaze > 1
x_bool2 = x_gaze < 0 
y_bool1 = y_gaze > 1
y_bool2 = y_gaze < 0

x_gaze[x_bool1] = np.nan
x_gaze[x_bool2] = np.nan
y_gaze[y_bool1] = np.nan
y_gaze[y_bool2] = np.nan

x_gazePix = x_gaze * 1280
y_gazePix = y_gaze * 720

xy_gaze = [x_gaze, y_gaze]
xy_gazePix = [x_gazePix, y_gazePix]

left_timestamps = list(pupilR['pupil_timestamp'])
right_timestamps = list(pupilL['pupil_timestamp'])

new_lstart_idxs = []
new_rstart_idxs = []
new_lstop_idxs = []
new_rstop_idxs = []

undetected_df = pd.read_csv(SUBJECT+'undetected.csv',index_col=[0])
un_sts = list(undetected_df['Start Time'])
un_ets = list(undetected_df['Stop Time'])
un_types = list(undetected_df['Type'])

for i in range(len(un_sts)):
    diffs = np.absolute(np.array(left_timestamps) - un_sts[i])
    new_lstart_idxs.append((left_timestamps[list(diffs).index(min(diffs))],un_types[i]))
    diffs = np.absolute(np.array(right_timestamps) - un_sts[i])
    new_rstart_idxs.append((right_timestamps[list(diffs).index(min(diffs))],un_types[i]))
    
for i in range(len(un_ets)):
    diffs = np.absolute(np.array(left_timestamps) - un_ets[i])
    new_lstop_idxs.append((left_timestamps[list(diffs).index(min(diffs))], un_types[i]))
    diffs = np.absolute(np.array(right_timestamps) - un_ets[i])
    new_rstop_idxs.append((right_timestamps[list(diffs).index(min(diffs))],un_types[i]))

vel = vecvel(np.array([pupilR['angleX'], pupilR['angleY']]).T, 120, 2)
vel2 = vecvel(np.array([pupilL['angleX'], pupilL['angleY']]).T, 120, 2)
sac_1, radius_1 = microsacc2_nlp2(np.array([pupilR['angleX'], pupilR['angleY']]).T,vel,5,5)
sac_2, radius_2 = microsacc2_nlp2(np.array([pupilL['angleX'], pupilL['angleY']]).T,vel2,5,5)

index = []
for i in range(len(sac_1)):
    index_range = np.arange(sac_1[i][0], sac_1[i][1] + 1)
    index_diff = len(index_range)
    time_range = np.array(pupilR['pupil_timestamp'])[index_range]
    time_range2 = np.array(pupilR['pupil_timestamp'])[index_range + 1]

    diff = (time_range2 - time_range) >= 0.2
    
    if np.any(diff == 1):
        index.append(i)
        
sac_1 = np.delete(sac_1, index, axis=0)

index = []
for i in range(len(sac_2)):
    index_range = np.arange(sac_2[i][0], sac_2[i][1] + 1)
    
    time_range = np.array(pupilL['pupil_timestamp'])[index_range]
    time_range2 = np.array(pupilL['pupil_timestamp'])[index_range + 1]

    diff = (time_range2 - time_range) >= 0.2
    
    if np.any(diff == 1):
        index.append(i)

sac_2 = np.delete(sac_2, index, axis=0)

# Bino saccade detection
sac_1t = np.copy(sac_1)
eighth_col = []
ninth_col = []
for i in range(len(sac_1)):
    eighth_col.append(pupilR['pupil_timestamp'].iloc[int(sac_1t[i, 0])])
    ninth_col.append(pupilR['pupil_timestamp'].iloc[int(sac_1t[i, 1])])
    
eighth_col = np.array(eighth_col)
ninth_col = np.array(ninth_col)
sac_1t = np.concatenate((sac_1t, eighth_col[:, np.newaxis], ninth_col[:, np.newaxis]), axis=1)

sac_2t = np.copy(sac_2)
seventh_col = []
eighth_col = []
for i in range(len(sac_2)):
    seventh_col.append(pupilL['pupil_timestamp'].iloc[int(sac_2t[i,0])])
    eighth_col.append(pupilL['pupil_timestamp'].iloc[int(sac_2t[i,0])])

seventh_col = np.array(seventh_col)
eighth_col = np.array(eighth_col)
sac_2t = np.concatenate((sac_2t, seventh_col[:, np.newaxis], eighth_col[:, np.newaxis]), axis=1)

sac_bi = binsacc(sac_2, sac_1)
sac_bit = binsaccT(sac_2t, sac_1t)

r_saccade_start = pupilR.iloc[sac_bit[:, 7].astype(int), :]
r_saccade_end = pupilR.iloc[sac_bit[:, 8].astype(int), :]
r_angleX = r_saccade_end.iloc[:, 3] - r_saccade_start.iloc[:, 3]
r_angleY = r_saccade_end.iloc[:, 4] - r_saccade_start.iloc[:, 4]
r_angleXY = np.column_stack((r_angleX, r_angleY))

l_saccade_start = pupilL.iloc[sac_bit[:, 9].astype(int), :]
l_saccade_end = pupilL.iloc[sac_bit[:, 10].astype(int), :]
l_angleX = l_saccade_end.iloc[:, 3] - l_saccade_start.iloc[:, 3]
l_angleY = l_saccade_end.iloc[:, 4] - l_saccade_start.iloc[:, 4]
l_angleXY = np.column_stack((l_angleX, l_angleY))

diff_angleX = r_angleX - l_angleX
diff_angleY = r_angleY - l_angleY
diff_angleXY = np.column_stack((diff_angleX, diff_angleY))

colNames = ['pupil_timestamp(s)', 'mono/bino', 'eye(r/l)']

# bino saccades in right/left eye
bino_right = pd.DataFrame({'pupil_timestamp(s)': r_saccade_start.iloc[:, 0],
                           'mono/bino': 'bino',
                           'eye(r/l)': 'r'})


bino_left = pd.DataFrame({'pupil_timestamp(s)': l_saccade_start.iloc[:, 0],
                          'mono/bino': 'bino',
                          'eye(r/l)': 'l'})

# mono saccades in right/left eye
mono_right = pd.DataFrame({'pupil_timestamp(s)': sac_1t[:, 7],
                           'mono/bino': 'mono',
                           'eye(r/l)': 'r'})
mono_left = pd.DataFrame({'pupil_timestamp(s)': sac_2t[:, 7],
                          'mono/bino': 'mono',
                          'eye(r/l)': 'l'})

# combine into one big table
table = pd.concat([bino_right, bino_left, mono_right, mono_left])
table = table.sort_values(by=['pupil_timestamp(s)', 'mono/bino'])

def update_frame():
    if not paused:
        ret, frame = cap.read()  # Read a frame from the video capture

        if ret:
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            video_slider.set(current_frame/30)
            # Update the plots with new data
            update_plots(current_frame)

    # Schedule the next frame update
    delay = int(1000 / fps)  # Compute the delay based on the video's frame rate
    root.after(delay, update_frame)

def toggle_play():
    global paused

    if paused:
        paused = False
        play_pause_button.config(text="Pause")
    else:
        paused = True
        play_pause_button.config(text="Play")

def on_slider_change(value):
    value = round(float(value) * 30)
    global manual_adjustment
    ret, frame = cap.read() 
    # Update the video frame based on the slider value
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(value))
    
    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the frame to fit the Tkinter window
    frame_resized = cv2.resize(frame_rgb, (video_width, video_height))

    # Convert the resized frame to ImageTk format
    img = Image.fromarray(frame_resized)
    img_tk = ImageTk.PhotoImage(image=img)

    # Update the canvas with the new frame
    video_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    video_canvas.img = img_tk

    # Update the slider position if not being manually adjusted
    #if not manual_adjustment:
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    video_slider.set(current_frame/30)
    plot_slider.set(current_frame/30)
    
    # Update the plots with new data
    update_plots(current_frame)


    # Set manual_adjustment flag to indicate user adjustment
    manual_adjustment = True
    
def on_slider_change2(value):
    value = round(float(value) * 30)
    update_plots(value)
    
def update_plots(current_frame):
    # Clear previous plots
    plot_ax1.clear()
    plot_ax2.clear()
    
    current_time = current_frame / 30
    start_time = max([0, current_time-0.5])
    end_time = current_time+0.5
    
    r_idxs = [i for i, t in enumerate(list(pupilR['pupil_timestamp'])) if start_time <= t <= end_time]
    l_idxs = [i for i, t in enumerate(list(pupilL['pupil_timestamp'])) if start_time <= t <= end_time]
    # Plot new data
    if len(r_idxs) != 0 and len(l_idxs) != 0:
        plot_ax1.scatter(list(pupilR['pupil_timestamp'])[r_idxs[0]:r_idxs[-1]], list(pupilR['angleX'])[r_idxs[0]:r_idxs[-1]], s=8, color='red')
        plot_ax1.scatter(list(pupilL['pupil_timestamp'])[l_idxs[0]:l_idxs[-1]], list(pupilL['angleX'])[l_idxs[0]:l_idxs[-1]], s=8, color='blue')
        
        plot_ax2.scatter(list(pupilR['pupil_timestamp'])[r_idxs[0]:r_idxs[-1]], list(pupilR['angleY'])[r_idxs[0]:r_idxs[-1]], s=8, color='red')
        plot_ax2.scatter(list(pupilL['pupil_timestamp'])[l_idxs[0]:l_idxs[-1]], list(pupilL['angleY'])[l_idxs[0]:l_idxs[-1]], s=8, color='blue')
    
    plot_ax1.scatter([list(pupilR['pupil_timestamp'])[int(i)] for i in sac_1[:,0] if i in r_idxs and float(list(pupilR['pupil_timestamp'])[int(i)]) not in remove_times], [list(pupilR['angleX'])[int(i)] for i in sac_1[:,0] if i in r_idxs and float(list(pupilR['pupil_timestamp'])[int(i)]) not in remove_times], s=64, color='green')
    plot_ax1.scatter([list(pupilL['pupil_timestamp'])[int(i)] for i in sac_2[:,0] if i in r_idxs and float(list(pupilL['pupil_timestamp'])[int(i)]) not in remove_times], [list(pupilL['angleX'])[int(i)] for i in sac_2[:,0] if i in r_idxs and float(list(pupilL['pupil_timestamp'])[int(i)]) not in remove_times], s=64, color='green')
    plot_ax1.scatter([list(pupilR['pupil_timestamp'])[int(i)] for i in sac_bit[:,7] if i in r_idxs and float(list(pupilR['pupil_timestamp'])[int(i)]) not in remove_times], [list(pupilR['angleX'])[int(i)] for i in sac_bit[:,7] if i in r_idxs and float(list(pupilR['pupil_timestamp'])[int(i)]) not in remove_times], s=96, color='blue')
    plot_ax1.scatter([list(pupilL['pupil_timestamp'])[int(i)] for i in sac_bit[:,9] if i in r_idxs and float(list(pupilL['pupil_timestamp'])[int(i)]) not in remove_times], [list(pupilL['angleX'])[int(i)] for i in sac_bit[:,9] if i in r_idxs and float(list(pupilL['pupil_timestamp'])[int(i)]) not in remove_times], s=96, color='red')
    plot_ax1.scatter([list(pupilR['pupil_timestamp'])[int(i)] for i in sac_1[:,1] if i in r_idxs and float(list(pupilR['pupil_timestamp'])[int(i)]) not in remove_times], [list(pupilR['angleX'])[int(i)] for i in sac_1[:,1] if i in r_idxs and float(list(pupilR['pupil_timestamp'])[int(i)]) not in remove_times], s=64, color='red')
    plot_ax1.scatter([list(pupilL['pupil_timestamp'])[int(i)] for i in sac_2[:,1] if i in r_idxs and float(list(pupilL['pupil_timestamp'])[int(i)]) not in remove_times], [list(pupilL['angleX'])[int(i)] for i in sac_2[:,1] if i in r_idxs and float(list(pupilL['pupil_timestamp'])[int(i)]) not in remove_times], s=64, color='red')
    plot_ax1.scatter([list(pupilR['pupil_timestamp'])[int(i)] for i in sac_bit[:,8] if i in r_idxs and float(list(pupilR['pupil_timestamp'])[int(i)]) not in remove_times], [list(pupilR['angleX'])[int(i)] for i in sac_bit[:,8] if i in r_idxs and float(list(pupilR['pupil_timestamp'])[int(i)]) not in remove_times], s=96, color='blue')
    plot_ax1.scatter([list(pupilL['pupil_timestamp'])[int(i)] for i in sac_bit[:,10] if i in r_idxs and float(list(pupilL['pupil_timestamp'])[int(i)]) not in remove_times], [list(pupilL['angleX'])[int(i)] for i in sac_bit[:,10] if i in r_idxs and float(list(pupilL['pupil_timestamp'])[int(i)]) not in remove_times], s=96, color='red')

    plot_ax2.scatter([list(pupilR['pupil_timestamp'])[int(i)] for i in sac_1[:,0] if i in l_idxs and float(list(pupilR['pupil_timestamp'])[int(i)]) not in remove_times], [list(pupilR['angleY'])[int(i)] for i in sac_1[:,0] if i in l_idxs and float(list(pupilR['pupil_timestamp'])[int(i)]) not in remove_times], s=64, color='green')
    plot_ax2.scatter([list(pupilL['pupil_timestamp'])[int(i)] for i in sac_2[:,0] if i in l_idxs and float(list(pupilL['pupil_timestamp'])[int(i)]) not in remove_times], [list(pupilL['angleY'])[int(i)] for i in sac_2[:,0] if i in l_idxs and float(list(pupilR['pupil_timestamp'])[int(i)]) not in remove_times], s=64, color='green')
    plot_ax2.scatter([list(pupilR['pupil_timestamp'])[int(i)] for i in sac_bit[:,7] if i in l_idxs and float(list(pupilR['pupil_timestamp'])[int(i)]) not in remove_times], [list(pupilR['angleY'])[int(i)] for i in sac_bit[:,7] if i in l_idxs and float(list(pupilR['pupil_timestamp'])[int(i)]) not in remove_times], s=96, color='blue')
    plot_ax2.scatter([list(pupilL['pupil_timestamp'])[int(i)] for i in sac_bit[:,9] if i in l_idxs and float(list(pupilL['pupil_timestamp'])[int(i)]) not in remove_times], [list(pupilL['angleY'])[int(i)] for i in sac_bit[:,9] if i in l_idxs and float(list(pupilR['pupil_timestamp'])[int(i)]) not in remove_times], s=96, color='red')
    plot_ax2.scatter([list(pupilR['pupil_timestamp'])[int(i)] for i in sac_1[:,1] if i in l_idxs and float(list(pupilR['pupil_timestamp'])[int(i)]) not in remove_times], [list(pupilR['angleY'])[int(i)] for i in sac_1[:,1] if i in l_idxs and float(list(pupilR['pupil_timestamp'])[int(i)]) not in remove_times], s=64, color='red')
    plot_ax2.scatter([list(pupilL['pupil_timestamp'])[int(i)] for i in sac_2[:,1] if i in l_idxs and float(list(pupilL['pupil_timestamp'])[int(i)]) not in remove_times], [list(pupilL['angleY'])[int(i)] for i in sac_2[:,1] if i in l_idxs and float(list(pupilR['pupil_timestamp'])[int(i)]) not in remove_times], s=64, color='red')
    plot_ax2.scatter([list(pupilR['pupil_timestamp'])[int(i)] for i in sac_bit[:,8] if i in l_idxs and float(list(pupilR['pupil_timestamp'])[int(i)]) not in remove_times], [list(pupilR['angleY'])[int(i)] for i in sac_bit[:,8] if i in l_idxs and float(list(pupilR['pupil_timestamp'])[int(i)]) not in remove_times], s=96, color='blue')
    plot_ax2.scatter([list(pupilL['pupil_timestamp'])[int(i)] for i in sac_bit[:,10] if i in l_idxs and float(list(pupilL['pupil_timestamp'])[int(i)]) not in remove_times], [list(pupilL['angleY'])[int(i)] for i in sac_bit[:,10] if i in l_idxs and float(list(pupilR['pupil_timestamp'])[int(i)]) not in remove_times], s=96, color='red')

    for i in range(len(new_lstart_idxs)):
        plot_ax1.scatter(left_timestamps[int(new_lstart_idxs[i][0])], list(pupilL['angleX'])[int(new_lstart_idxs[i][0])], s=96, color='green', marker='D')
        plot_ax1.scatter(left_timestamps[int(new_lstop_idxs[i][0])], list(pupilL['angleX'])[int(new_lstop_idxs[i][0])], s=96, color='red', marker='D')
        plot_ax2.scatter(left_timestamps[int(new_lstart_idxs[i][0])], list(pupilL['angleY'])[int(new_lstart_idxs[i][0])], s=96, color='green', marker='D')
        plot_ax2.scatter(left_timestamps[int(new_lstop_idxs[i][0])], list(pupilL['angleY'])[int(new_lstop_idxs[i][0])], s=96, color='red', marker='D')

    for i in range(len(new_lstop_idxs)):
        plot_ax1.scatter(right_timestamps[int(new_rstart_idxs[i][0])], list(pupilR['angleX'])[int(new_rstart_idxs[i][0])], s=96, color='green', marker='D')
        plot_ax1.scatter(right_timestamps[int(new_rstop_idxs[i][0])], list(pupilR['angleX'])[int(new_rstop_idxs[i][0])], s=96, color='red', marker='D')
        plot_ax2.scatter(right_timestamps[int(new_rstart_idxs[i][0])], list(pupilR['angleY'])[int(new_rstart_idxs[i][0])], s=96, color='green', marker='D')
        plot_ax2.scatter(right_timestamps[int(new_rstop_idxs[i][0])], list(pupilR['angleY'])[int(new_rstop_idxs[i][0])], s=96, color='red', marker='D')
    
    plot_ax1.plot([current_time, current_time], [-90, 90], 'k')
    plot_ax2.plot([current_time, current_time], [-90, 90], 'k')
    
    plot_ax1.set_title('X PUPIL Saccades')
    plot_ax2.set_title('Y PUPIL Saccades')
    plot_ax1.set_xlabel('Time (seconds)')
    plot_ax2.set_xlabel('Time (seconds)')
    plot_ax1.set_ylabel('X Angle position (degrees)')
    plot_ax2.set_ylabel('Y Angle position (degrees)')
    plot_ax1.set_xlim((start_time, end_time))
    plot_ax2.set_xlim((start_time, end_time))
    plot_ax1.set_ylim((-90,90))
    plot_ax2.set_ylim((-90,90))
    plot_ax1.set_xticks(np.arange(max([0,round((current_time*10)-(0.5*10))/10]), round((current_time*10)+(0.5*10))/10, step=0.1))
    plot_ax2.set_xticks(np.arange(max([0,round((current_time*10)-(0.5*10))/10]), round((current_time*10)+(0.5*10))/10, step=0.1))

    # plot_ax1.plot(pupilR['pupil_timestamp'], pupilR['angleX'], 'o', markersize=8, markeredgecolor='green', markerfacecolor='green', color='blue', linewidth=1)
    # Update the plot canvases
    plot_canvas1.draw()
    plot_canvas2.draw()

# Create a Tkinter window
root = tk.Tk()
root.title("Video Player and Plots")

# Open a video file
video_path = SUBJECT+"world.mp4"
cap = cv2.VideoCapture(video_path)

# Get the video properties
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/1.3)
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/1.3)
fps = 30
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create a frame for the video display
video_frame = tk.Frame(root)
video_frame.grid(row=0, column=0)

# Create a canvas to display the video frames
video_canvas = tk.Canvas(video_frame, width=video_width, height=video_height)
video_canvas.pack()

# Create a slider for adjusting the frame number
video_slider = tk.Scale(root, from_=0.0, to=(num_frames-1)/30, length=800, resolution = 1/30, tickinterval=10, orient=tk.HORIZONTAL, command=on_slider_change)
video_slider.grid(row=1, column=0, pady=1)

# Create a slider for adjusting plot position
plot_slider = tk.Scale(root, from_=0.0, to=(num_frames-1)/30, length=400, resolution = 1/30, orient=tk.HORIZONTAL, command=on_slider_change2)
plot_slider.grid(row=1, column=1, pady=1)

# Create a button for playing and pausing the video
paused = False
play_pause_button = tk.Button(root, text="Pause", command=toggle_play)
play_pause_button.grid(row=2, column=0, pady=1)
'''
prev_button = tk.Button(root, text="Prev", command=go_prev)
prev_button.grid(row=2, column=1, pady=15, sticky='w')

next_button = tk.Button(root, text="Next", command=go_next)
next_button.grid(row=2, column=1, pady=15)
'''
# Create a frame for the upper plot display
plot_frame1 = tk.Frame(root, height=5, pady=1)
plot_frame1.grid(row=0, column=1, padx=1, pady=1, sticky="n")

# Create a figure and axis for the upper plot
plot_fig1 = plt.figure(figsize=(3.9,2.7))
plot_ax1 = plot_fig1.add_subplot(111)

# Create a canvas to display the upper plot
plot_canvas1 = FigureCanvasTkAgg(plot_fig1, master=plot_frame1)
plot_canvas1.draw()
plot_canvas1.get_tk_widget().pack()

# Create a frame for the lower plot display
plot_frame2 = tk.Frame(root, height=5, pady=1)
plot_frame2.grid(row=0, column=1, padx=1, pady=1, sticky='s')

# Create a figure and axis for the lower plot
plot_fig2 = plt.figure(figsize=(3.9,2.7))
plot_ax2 = plot_fig2.add_subplot(111)

# Create a canvas to display the lower plot
plot_canvas2 = FigureCanvasTkAgg(plot_fig2, master=plot_frame2)
plot_canvas2.draw()
plot_canvas2.get_tk_widget().pack()

# Flag to track manual adjustment of the slider
manual_adjustment = False

# Start updating the frames
update_frame()

# Run the Tkinter event loop
root.mainloop()

# Release the video capture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()