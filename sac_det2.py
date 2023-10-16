import tkinter as tk
import cv2
from tkinter import ttk
from PIL import ImageTk, Image
import pandas as pd
import os
import numpy as np
import sys

SUBJECT = sys.argv[1]
#'F:\InfantSceneData(3_22_23)/2023_07_13/28277/exports_adult/000/'

# txt file containing detection times
MATRIX = SUBJECT + 'matrix.txt'
# csv file that stores times where the eyes aren't being tracked
NO_TRACKING = SUBJECT + 'no_tracking.csv'
# csv file that stores detection times and scoring
SCORES = SUBJECT + 'scores.csv'
# csv file that stores detections not found by automated detector
UNDETECTED = SUBJECT + 'undetected.csv'
# World video
WORLD_VID = SUBJECT + "world.mp4"
# Plot video
PLOT_VID = SUBJECT + "plot_video.avi"

untrack_times = [0, 0]
undetected_times = [0,0]
time_idxs = []
df = pd.read_csv(MATRIX)
world_vid = cv2.VideoCapture(WORLD_VID)
FPS = round(world_vid.get(cv2.CAP_PROP_FPS))

if 'no_tracking.csv' not in os.listdir(SUBJECT):
    nt_df = pd.DataFrame({'Start Time': [], 'Stop Time': [], 'Eye': []})
    nt_df.to_csv(NO_TRACKING)
    
if 'scores.csv' not in os.listdir(SUBJECT):
    score_df = pd.DataFrame({'Time': [], 'Score':[], 'Eye': []})
    score_df.to_csv(SCORES)
    
if 'undetected.csv' not in os.listdir(SUBJECT):
    undetected_df = pd.DataFrame({'Start Time': [], 'Stop Time': [], 'Type': []})
    undetected_df.to_csv(UNDETECTED)

for i in range(len(df)):
    time_idxs.append(float(df.iloc[i]))

time_idxs.append(float(df.columns[0]))
temp = []
for t in time_idxs:
    if t not in temp:
        temp.append(t)
time_idxs = temp
time_idxs = list(sorted(time_idxs))
rev_times = list(reversed(time_idxs))

def update_frame():
    if not paused:
        ret, frame = cap.read()  # Read a frame from the video capture

        if ret:
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            video_slider.set(current_frame/FPS)
        
    # Schedule the next frame update
    delay = int(1000 / fps)  # Compute the delay based on the video's frame rate
    root.after(delay, update_frame)

def toggle_speed(value):
    global fps
    if value == 'Normal':
        fps=FPS
    if value == 'Slow x2':
        fps=1
    if value == 'Slow':
        fps=5
        
def change_untrack_menu(value):
    eye_untrack_var.set(value)
    
def change_good_menu(value):
    eye_good_var.set(value)
        
def find_first_lower(sorted_list, target):
    left, right = 0, len(sorted_list) - 1

    while left <= right:
        mid = (left + right) // 2
        if sorted_list[mid] < target:
            # The current element is lower than the target.
            # Check if the previous element is higher or equal to the target.
            if mid == len(sorted_list) - 1 or sorted_list[mid + 1] >= target:
                return sorted_list[mid]
            else:
                left = mid + 1
        else:
            # The current element is higher or equal to the target.
            right = mid - 1

    return None  # If no element in the list is lower than the target.
        
def go_prev():
    global time

    if time < time_idxs[0]:
        pass
    else:
        score_df = pd.read_csv(SCORES,index_col=[0])
        score_times = list(score_df['Time'])
        for t in rev_times:
            if t < time:
                time = t
                video_slider.set(t)
                return
        
def go_next():
    global time

    if time > time_idxs[-1]:
        pass
    else:
        score_df = pd.read_csv(SCORES,index_col=[0])
        score_times = list(score_df['Time'])
        for t in time_idxs:
            if t > time:
                time = t
                video_slider.set(t)
                return
            
def toggle_play():
    global paused

    if paused:
        paused = False
        play_pause_button.config(text="Pause")
    else:
        paused = True
        play_pause_button.config(text="Play")

def track_untrackable():
    global time
    global untrackable
    global untrack_times
    
    if not untrackable:
        untrackable = True
        untrack_times[0] = time
        nt_button.config(text="Stop Recording")
    else:
        untrackable = False
        untrack_times[1] = time
        nt_df = pd.read_csv(NO_TRACKING, index_col=[0])
        sts = list(nt_df['Start Time'])
        ets = list(nt_df['Stop Time'])
        eyes = list(nt_df['Eye'])
        sts.append(untrack_times[0])
        ets.append(untrack_times[1])
        eyes.append(eye_untrack_var.get())
        ntlist.insert(tk.END, "{} to {}: {}".format(str(round(sts[-1],2)),str(round(ets[-1],2)), eyes[-1]))
        nt_df = pd.DataFrame({'Start Time': sts, 'Stop Time': ets, 'Eye': eyes})
        nt_df.to_csv(NO_TRACKING)
        nt_button.config(text="Record Untrackable")
        
def undetected_ss():
    global time
    global undetected
    global undetected_times
    
    if not undetected:
        undetected = True
        undetected_times[0] = time
        undetected_button.config(text='Stop Undetected')
    else:
        undetected = False
        undetected_times[1] = time
        undetected_df = pd.read_csv(UNDETECTED, index_col=[0])
        un_sts = list(undetected_df['Start Time'])
        un_ets = list(undetected_df['Stop Time'])
        un_types = list(undetected_df['Type'])
        un_sts.append(undetected_times[0])
        un_ets.append(undetected_times[1])
        un_types.append(eye_movement_var.get())
        undetectedlist.insert(tk.END, "{} to {}: {}".format(str(round(un_sts[-1],2)),str(round(un_ets[-1],2)),un_types[-1]))
        undetected_df = pd.DataFrame({'Start Time': un_sts, 'Stop Time': un_ets, 'Type': un_types})
        undetected_df.to_csv(UNDETECTED)
        undetected_button.config(text='Record Undetected')
        
def delete_untrackable():
    del_item = ntlist.curselection()
    if len(del_item) == 0:
        return
    ntlist.delete(del_item[0])
    nt_df = pd.read_csv(NO_TRACKING,index_col=[0])
    sts = list(nt_df['Start Time'])
    ets = list(nt_df['Stop Time'])
    eyes = list(nt_df['Eye'])
    del sts[del_item[0]]
    del ets[del_item[0]]
    del eyes[del_item[0]]
    nt_df = pd.DataFrame({'Start Time': sts, 'Stop Time': ets, 'Eye': eyes})
    nt_df.to_csv(NO_TRACKING)
    
def undetected_jump():
    jump_idx = undetectedlist.curselection()
    if len(jump_idx) == 0:
        return
    jump_idx = jump_idx[0]
    jump_time = undetectedlist.get(jump_idx)
    jump_time = jump_time.split(' ')
    jump_time = float(jump_time[0])
    video_slider.set(jump_time)

    
def delete_undetected():
    del_item = undetectedlist.curselection()
    if len(del_item) == 0:
        return
    undetectedlist.delete(del_item[0])
    undetected_df = pd.read_csv(UNDETECTED,index_col=[0])
    sts = list(undetected_df['Start Time'])
    ets = list(undetected_df['Stop Time'])
    types = list(undetected_df['Type'])
    del sts[del_item[0]]
    del ets[del_item[0]]
    del types[del_item[0]]
    undetected_df = pd.DataFrame({'Start Time': sts, 'Stop Time': ets, 'Type': types})
    undetected_df.to_csv(UNDETECTED)
    
def score():
    global time
    global paused

    pred = score_var.get()
    score_df = pd.read_csv(SCORES, index_col=[0])
    score_times = list(score_df['Time'])
    scores = list(score_df['Score'])
    eyes = list(score_df['Eye'])
    score_times.append(plot_slider.get())
    if pred == 'Correct':
        scores.append(1)
        eyes.append(eye_good_var.get())
        scorelist.insert(tk.END, "{}: {} ({})".format(str(round(time,2)), str(1), eyes[-1]))
    if pred == 'Incorrect':
        scores.append(0)
        eyes.append(eye_good_var.get())
        scorelist.insert(tk.END, "{}: {} ({})".format(str(round(time,2)), str(0), eyes[-1]))
    scores_df = pd.DataFrame({'Time': score_times, 'Score': scores, 'Eye': eyes})
    scores_df.to_csv(SCORES)
    score_acc_var.set('Accuracy: {}'.format(str(round(sum(scores)/len(scores),2))))
    score_acc_lab['textvariable']=score_acc_var
    score_perc_var.set('Percentage Scored: {}'.format(round(len(scores)/len(time_idxs),2)))
    score_perc_lab['textvariable']=score_perc_var
    
def delete_score():
    del_item = scorelist.curselection()
    if len(del_item) == 0:
        return
    scorelist.delete(del_item[0])
    score_df = pd.read_csv(SCORES, index_col=[0])
    score_times = list(score_df['Time'])
    scores = list(score_df['Score'])
    eyes = list(score_df['Eye'])
    del score_times[del_item[0]]
    del scores[del_item[0]]
    score_df = pd.DataFrame({'Time': score_times, 'Score': scores, 'Eye': eyes})
    score_df.to_csv(SCORES)
    if len(scores) == 0:
        score_acc_var.set('Accuracy: 0')
        score_acc_lab['textvariable']=score_acc_var
        score_perc_var.set('Percentage Scored: 0')
        score_perc_lab['textvariable']=score_perc_var
        return
    score_acc_var.set('Accuracy: {}'.format(str(round(sum(scores)/len(scores),2))))
    score_acc_lab['textvariable']=score_acc_var
    score_perc_var.set('Percentage Scored: {}'.format(round(len(scores)/len(time_idxs),2)))
    score_perc_lab['textvariable']=score_perc_var
    
def change_menu(value):
    score_var.set(value)

def em_menu(value):
    eye_movement_var.set(value)

def on_slider_change2(value):
    value = round(float(value)*FPS)
    ret, frame = cap2.read()
    cap2.set(cv2.CAP_PROP_POS_FRAMES, int(value))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    zoomed_frame = cv2.resize(frame_rgb, None, fx=zoom_level, fy=zoom_level)
    
    img = Image.fromarray(zoomed_frame)
    img_tk = ImageTk.PhotoImage(image=img)
    
    video_canvas2.create_image(0, 0, anchor=tk.NW, image=img_tk)
    video_canvas2.img = img_tk
    
    video_canvas2.bind('<ButtonPress-1>', start_drag)
    video_canvas2.bind('<B1-Motion>', drag)
    video_canvas2.bind('<ButtonRelease-1>', end_drag)
    
def start_drag(event):
    global dragging
    dragging = True
    video_canvas2.scan_mark(event.x, event.y)

def drag(event):
    if dragging:
        video_canvas2.scan_dragto(event.x, event.y, gain=1)
        
def end_drag(event):
    global dragging
    dragging = False
    
def zoom_in():
    global zoom_level
    zoom_level *= 1.1
    if zoom_level > 2.0:
        zoom_level = 2.0
    on_slider_change2(plot_slider.get())
    
def zoom_out():
    global zoom_level
    zoom_level /= 1.1
    if zoom_level < 0.9:
        zoom_level = 0.9
    on_slider_change2(plot_slider.get())

def on_slider_change(value):
    value = round(float(value) * FPS)
    global manual_adjustment
    global time
    #nt_df = pd.read_csv(NO_TRACKING, index_col=[0])
    #nt = np.array(nt_df)
    ret, frame = cap.read()
    if ret:
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
        
        time = current_frame/FPS
        video_slider.set(time)

    
    ret2, frame2 = cap2.read() 
    if ret2:
        # Update the video frame based on the slider value
        cap2.set(cv2.CAP_PROP_POS_FRAMES, int(value))
        
        # Convert the frame to RGB format
        frame_rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    
        zoomed_frame = cv2.resize(frame_rgb2, None, fx=zoom_level, fy=zoom_level)
    
        # Convert the resized frame to ImageTk format
        img2 = Image.fromarray(zoomed_frame)
        img_tk2 = ImageTk.PhotoImage(image=img2)
    
        # Update the canvas with the new frame
        video_canvas2.create_image(0, 0, anchor=tk.NW, image=img_tk2)
        video_canvas2.img = img_tk2
    
        current_frame = int(cap2.get(cv2.CAP_PROP_POS_FRAMES))
        time = current_frame/FPS
        plot_slider.set(time)
        
        video_canvas2.bind('<ButtonPress-1>', start_drag)
        video_canvas2.bind('<B1-Motion>', drag)
        video_canvas2.bind('<ButtonRelease-1>', end_drag)

    # Set manual_adjustment flag to indicate user adjustment
    manual_adjustment = True

# Create a Tkinter window
root = tk.Tk()

root.title("Video Player and Plots")
root.configure()
# Open a video file
video_path = WORLD_VID
cap = cv2.VideoCapture(video_path)

zoom_level =.9
dragging = False

# Get the video properties
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set the window size to match the monitor size
window_width = int(screen_width * 0.5)  # You can adjust the factor as needed
window_height = int(screen_height * 0.5)  # You can adjust the factor as needed

video_width = window_width #int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/1.2)
video_height = window_height #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/1.2)
fps = FPS
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create a frame for the video display
video_frame = ttk.Frame(root)
video_frame.grid(row=0, column=0, sticky='w')

# Create a canvas to display the video frames
video_canvas = tk.Canvas(video_frame, width=video_width, height=video_height)
video_canvas.pack()

# Open a video file
video_path = PLOT_VID
cap2 = cv2.VideoCapture(video_path)

# Get the video properties
video_width2 = window_width #int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)/1.2)
video_height2 = window_height
fps = FPS
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create a frame for the video display
video_frame2 = tk.Frame(root)
video_frame2.grid(row=0, column=1, sticky='w')

# Create a canvas to display the video frames
video_canvas2 = tk.Canvas(video_frame2, width=video_width2, height=video_height2)
video_canvas2.pack()

# Create a slider for adjusting the frame number
video_slider = tk.Scale(root, from_=0.0, to=(num_frames-1)/FPS, length=video_width, resolution = 1/FPS, tickinterval=10, orient=tk.HORIZONTAL,command=on_slider_change)
video_slider.grid(row=1, column=0, pady=1, sticky='w')

plot_slider = tk.Scale(root, from_=0.0, to=(num_frames-1)/FPS, length=video_width2, resolution = 1/FPS, orient=tk.HORIZONTAL,command=on_slider_change2)
plot_slider.grid(row=1, column=1, padx=20, sticky='w')

# Zoom buttons
zoom_in_button = ttk.Button(root, text="Zoom In", command=zoom_in)
zoom_in_button.grid(row=6, column=1, padx=1, sticky='w')

zoom_out_button = ttk.Button(root, text="Zoom Out", command=zoom_out)
zoom_out_button.grid(row=6, column=1, padx=80, sticky='w')

# Create a button for playing and pausing the video
paused = False
play_pause_button = ttk.Button(root, text="Pause", command=toggle_play)
play_pause_button['style'] = 'Emergency.TButton'
play_pause_button.grid(row=2, column=0)

untrackable = False
nt_button = ttk.Button(root, text='Record Untrackable',command=track_untrackable)
nt_button['style'] = 'Emergency.TButton'
nt_button.grid(row=3, column=0, padx=1, sticky='w')

delete_nt_button = ttk.Button(root, text='Delete Untrackable', command=delete_untrackable)
delete_nt_button['style'] = 'Emergency.TButton'
delete_nt_button.grid(row=6, column=0, padx=1, sticky='w')

prev_button = ttk.Button(root, text='Prev',  command=go_prev)
prev_button['style'] = 'Emergency.TButton'
prev_button.grid(row=2, column=1, padx=1, sticky='w')

next_button = ttk.Button(root, text='Next',command=go_next)
next_button['style'] = 'Emergency.TButton'
next_button.grid(row=2, column=1, padx=1)

variable = tk.StringVar(root)
variable.set("Normal") # default value

w = tk.OptionMenu(root, variable, "Normal", "Slow x2", "Slow", command=toggle_speed)
w.grid(row=3, column=0)

# Flag to track manual adjustment of the slider
manual_adjustment = False

nt_title = 'Times with no eye tracking:'

nt_var = tk.StringVar(root)
nt_var.set(nt_title)

nt_lab = tk.Label(root,textvariable=nt_var)
nt_lab.grid(row=4,column=0,padx=1,sticky='w')

nt_df = pd.read_csv(NO_TRACKING, index_col=[0])
sts = list(nt_df['Start Time'])
ets = list(nt_df['Stop Time'])
eyes = list(nt_df['Eye'])

ntlist = tk.Listbox(root, height=4 )
for i in range(len(sts)):
   ntlist.insert(tk.END, "{} to {}: {}".format(str(round(sts[i],2)),str(round(ets[i],2)), eyes[i]))

ntlist.grid(row=5,column=0,sticky='w')

score_title = "Prediction Scores (1 = Correct, 0 = Incorrect):"
score_var = tk.StringVar(root)
score_var.set(score_title)

score_lab = tk.Label(root,textvariable=score_var)
score_lab.grid(row=4,column=0,sticky='e')

score_df = pd.read_csv(SCORES, index_col=[0])
score_times = list(score_df['Time'])
scores = list(score_df['Score'])
score_eyes = list(score_df['Eye'])

scorelist = tk.Listbox(root, height=4)
for i in range(len(score_times)):
    scorelist.insert(tk.END, "{}: {} ({})".format(str(round(score_times[i],2)), str(scores[i]), score_eyes[i]))

scorelist.grid(row=5,column=0,sticky='e')

score_var = tk.StringVar(root)
score_var.set("Correct") # default value

score_menu = tk.OptionMenu(root, score_var, "Correct", "Incorrect", command=change_menu)
score_menu.grid(row=3, column=1, padx=1, sticky='sw')

score_button = ttk.Button(root, text='Score',command=score)
score_button['style'] = 'Emergency.TButton'
score_button.grid(row=3,column=0,sticky='e')

del_score_button = ttk.Button(root, text='Delete Score',command=delete_score)
del_score_button['style'] = 'Emergency.TButton'
del_score_button.grid(row=6, column=0, sticky='e')

if len(score_df) == 0:
    acc = 1
else:
    acc = sum(scores)/len(scores)
score_acc = "Accuracy: {}".format(str(round(acc,2)))

score_acc_var = tk.StringVar(root)
score_acc_var.set(score_acc)

score_acc_lab = tk.Label(root,textvariable=score_acc_var)
score_acc_lab.grid(row=4,column=1,sticky='nw')

if len(score_df) == 0:
    perc = 0
else:
    perc = len(score_df)/len(time_idxs)
score_perc = "Percentage Scored: {}".format(str(round(perc,2)))

score_perc_var = tk.StringVar(root)
score_perc_var.set(score_perc)

score_perc_lab = tk.Label(root, textvariable=score_perc_var)
score_perc_lab.grid(row=5,column=1,sticky='nw')

undetected = False

eye_movement_var = tk.StringVar(root)
eye_movement_var.set("Saccade") # default value

eye_movement_menu = tk.OptionMenu(root, eye_movement_var, "Saccade", "Pursuit", "Gaze", "Fixation", command=em_menu)
eye_movement_menu.grid(row=4, column=0, padx=int(video_width/2)-175, sticky='w')

undetected_button = ttk.Button(root, text='Record Undetected',command=undetected_ss)
undetected_button['style'] = 'Emergency.TButton'
undetected_button.grid(row=4, column=0)

undetected_df = pd.read_csv(UNDETECTED, index_col=[0])
undetected_sts = list(undetected_df['Start Time']) 
undetected_ets = list(undetected_df['Stop Time'])
undetected_types = list(undetected_df['Type'])

undetectedlist = tk.Listbox(root, height=4)
for i in range(len(undetected_sts)):
   undetectedlist.insert(tk.END, "{} to {}: {}".format(str(round(undetected_sts[i],2)),str(round(undetected_ets[i],2)),undetected_types[i]))

undetectedlist.grid(row=5,column=0)

del_undetected_button = ttk.Button(root, text='Delete Undetected',command=delete_undetected)
del_undetected_button['style'] = 'Emergency.TButton'
del_undetected_button.grid(row=6, column=0, padx=400, sticky='w')

jump_time_button = ttk.Button(root, text='Jump to Undetected', command=undetected_jump)
jump_time_button['style'] = 'Emergency.TButton'
jump_time_button.grid(row=6, column=0)

eye_untrack_var = tk.StringVar(root)
eye_untrack_var.set("Both")

eye_untrack_menu = tk.OptionMenu(root, eye_untrack_var, "Both", "Right", "Left", command=change_untrack_menu)
eye_untrack_menu.grid(row=3, column=0, padx=120, sticky='w')

eye_good_var = tk.StringVar(root)
eye_good_var.set("Both")

eye_good_menu = tk.OptionMenu(root, eye_good_var, "Both", "Right", "Left", command=change_good_menu)
eye_good_menu.grid(row=3, column=1, sticky='n')

# Start updating the frames
update_frame()

# Run the Tkinter event loop
root.columnconfigure(0, weight=1)
root.mainloop()

# Release the video capture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()