# Setup

**NOTE:** Must have [Anaconda](https://www.anaconda.com/download) and Python installed.

Open terminal (Mac) or anaconda prompt (Windows) and complete the following steps.

## Clone GitHub Repo

```
git clone https://github.com/ZachPetroff/saccade_detector.git
```

## Create Virtual Environment

```
conda create --name saccade_detector
conda activate saccade_detector
pip install -r requirements.txt
```

# Creating Plot Video in MATLAB

1. Open the MATLAB file 'plot_video.m'.
2. Update the paths on lines 25-27, 239, and 244 to match the subject code and date.
3. Click 'Run'.
4. After running the script, wait about 10 seconds for a 'Continue' button to appear. Click it to view the video in a separate window.
    - Do not close or resize the window, as it may cause an error.
    - The program is finished when the video stops playing. At that point, you can close the window and MATLAB.
    - If you continue working on the same video another day, you can skip reloading the video through MATLAB.

# Running GUI

- Open terminal (Mac) or anaconda prompt (Windows).

- Navigate to saccade detector folder:
       
    ```
    cd saccade_detector
    ```

- Activate virtual environment:
   
    ```
    conda activate sac_det
    ```
    
- Run GUI code (**update path before running**):
   
    ```
    python sac_det2.py 2023_03_27/khfn/exports/000/ 
    ```
        
- To view changes to the plot based on annotations, repeat steps a-c, then type `dynamic_plot_script.py` and press Enter.

## Coding Tools

### Cameras:
- World camera: Shows the subject's view and head movements.
- Right eye pupil tracker: Red dot in the center with a dark blue outer circle.
- Left eye pupil tracker: Red dot in the center with a dark blue outer circle.

### Graphs:
- Top "Y pupil" graph: Vertical eye movements.
- Bottom "X pupil" graph: Horizontal eye movements.
- Red line: Right eye.
- Blue line: Left eye.
- Red and blue lines running parallel indicate binocular data.

## Interpreting Eye Movements

1. Start at the beginning of the video on the slow setting.
2. Observe the red and blue graphs on the right while watching the world dot in the video.
3. Code movements based on the following criteria:

   - Fixation: Dot stays still, and the world camera is not moving. Red and blue lines should be parallel.
   - VOR (Vestibulo-ocular reflex): Dot stays still, but the world camera is moving. Red and blue lines should slope.
   - Saccade: Dot jumps from point A to point B, and the red and blue lines slope. Use the buttons under the graph to score saccades.

   - If the saccade was detected binocularly, select 'Both.'
   - If only one eye was detected, select 'Right' or 'Left' accordingly.
   - You can jump to the next predicted saccade using the 'Next' button.

   - Undetected Saccade: If a small saccade was missed, record it as an undetected saccade in the center.
   - Camera or Tracking Error: If the dot and red/blue lines disappear, it's likely an error. Click the 'record untrackable' button.
   - Lost Tracking with Data: If the dot disappears but the red/blue lines still provide data, grade the movement based on the graphs.
   - Vergence Movement: If the red and blue lines cross horizontally, it's likely a vergence movement.

### For More Information, Open the GUI_saccade PDF
