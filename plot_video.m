close all;
clearvars;
dbstop if error
all_fig = findall(0, 'type', 'figure');
close(all_fig)

%COLORS
RED       = [0.90 0.00 0.15 ];
MAGENTA   = [0.80 0.00 0.80 ];
ORANGE    = [1.00 0.50 0.00 ];
YELLOW    = [1.00 0.68 0.26 ];
GREEN     = [0.10 0.90 0.10 ];
CYAN      = [0.28 0.82 0.80 ];
BLUE      = [0.00 0.00 1.00 ];
BLACK     = [0.00 0.00 0.05 ];
GREY      = [0.46 0.44 0.42 ];
D_RED     = [0.63 0.07 0.18 ];
D_MAGENTA = [0.55 0.00 0.55 ];
D_ORANGE  = [1.00 0.30 0.10 ];
D_YELLOW  = [0.72 0.53 0.04 ];
D_GREEN   = [0.09 0.27 0.23 ];
D_BLUE    = [0.00 0.20 0.60 ];
D_CYAN    = [0.00 0.55 0.55 ];

gaze = readtable("C:\Users\Zachp\Downloads\same_itgd_2023_07_03_PL\exports_adult\000\gaze_positions.csv") %readtable('F:\InfantSceneData(3_22_23)/2023_02_07/czct/exports/000/gaze_positions.csv');
pupil = readtable("C:\Users\Zachp\Downloads\same_itgd_2023_07_03_PL\exports_adult\000\pupil_positions.csv") % readtable('F:\InfantSceneData(3_22_23)/2023_02_07/czct/exports/000/pupil_positions.csv');
vidObj = VideoReader("C:\Users\Zachp\Downloads\same_itgd_2023_07_03_PL\exports_adult\000\world.mp4") % VideoReader('F:\InfantSceneData(3_22_23)/2023_02_07/czct/exports/000/world.mp4');

gaze.gaze_timestamp = gaze.gaze_timestamp-gaze.gaze_timestamp(1);
pupil.pupil_timestamp = pupil.pupil_timestamp-pupil.pupil_timestamp(1);

%% find rows that contain less than some confidence and remove them
index = [];
for i  = 1:length(gaze.confidence)
%     if gaze.confidence(i) < 0.8
    if gaze.confidence(i) < 0.9
        index = [index i];
    end
end
gaze(index,:) = [];
disp(max(size(pupil)))
index = [];
for i  = 1:length(pupil.confidence)
%     if pupil.confidence(i) < 0.8
    if pupil.confidence(i) < 0.9
        index = [index i];
    end
end
pupil(index,:) = [];

% remove rows with nan's in pupil data
index = [];
for i  = 1:length(pupil.circle_3d_normal_x)
    if isnan(pupil.circle_3d_normal_x(i))
        index = [index i];
    end
end
pupil(index,:) = [];
disp(max(size(pupil)))
% converting pupil data into units of angle in degrees
for i = 1:length(pupil.pupil_timestamp)
    pupil.angleX(i) = atan(pupil.circle_3d_normal_x(i)/pupil.circle_3d_normal_z(i)) * (180/pi);
    pupil.angleY(i) = atan(pupil.circle_3d_normal_y(i)/pupil.circle_3d_normal_z(i)) * (180/pi);
end

% separate left and right eye pupil data
pupilR = pupil(pupil.eye_id == 0,:);
pupilL = pupil(pupil.eye_id == 1,:);

% flip signs of right eye as per Pupil Labs documentation
pupilR.angleX = pupilR.angleX * (-1);
pupilR.angleY = pupilR.angleY * (-1);

pupilRcomb = [pupilR.angleX pupilR.angleY];
pupilLcomb = [pupilL.angleX pupilL.angleY];

% assign gaze data to variables to be plugged into functions
x_gaze = gaze.norm_pos_x;
y_gaze = gaze.norm_pos_y;

% remove gaze data outside of the standardized 0 to 1 range (outside of
% world camera range)
%x_bool = x_gaze > 1 | x_gaze < 0;
%y_bool = y_gaze > 1 | y_gaze < 0;
%x_gaze(x_bool) = NaN;
%y_gaze(y_bool) = NaN;

x_gazePix = x_gaze * 1280; % pixel space
y_gazePix = y_gaze * 720;

xy_gaze = [x_gaze y_gaze]; % combined into 2 columns for undistortion later
xy_gazePix = [x_gazePix y_gazePix];

%% monocular saccade detection

% right and left eye velocity
vel  = vecvel([pupilR.angleX pupilR.angleY], 120,2);
vel2 = vecvel([pupilL.angleX pupilL.angleY], 120,2);

% detect saccades for each eye
[sac_1, radius_1] = ...
    microsacc2_nlp2([pupilR.angleX pupilR.angleY], vel  ,5,5); % 2-16-21 - 8,7 %started with 10,9 % Vel, Duration
[sac_2, radius_2] = ...
    microsacc2_nlp2([pupilL.angleX pupilL.angleY], vel2 ,5,5); % 2-16-21 - 8,7 %started with 10,9 % Vel, Duration

index = [];
for i = 1:length(sac_1)
    index_range = sac_1(i,1):1:sac_1(i,2); % steps in index
    index_diff = length(index_range); % how many samples in each saccade
    % NOW COMPARE THIS TO THE TIME RANGE AND SEE HOW MANY SAMPLES PER UNIT
    % TIME

    time_range = pupilR.pupil_timestamp(index_range);
    time_range2 = pupilR.pupil_timestamp(index_range+1);

    diff = (time_range2-time_range) >= 0.2; % look for difference >= 200 ms
    
    if any( diff == 1 )
        index = [index i];
    end
end
sac_1(index,:) = [];

index = [];
for i = 1:length(sac_2)
    index_range = sac_2(i,1):1:sac_2(i,2); % steps in index
    time_range = pupilL.pupil_timestamp(index_range);
    time_range2 = pupilL.pupil_timestamp(index_range+1);

    diff = (time_range2-time_range) >= 0.2; % look for difference >= 200 ms
    
    if any( diff == 1 )
        index = [index i];
    end
end
sac_2(index,:) = [];

%% Bino saccade detection

% maybe put change of index to time in here BEFORE going into temporal bino detector!
sac_1t = sac_1; %copies to retain original
for i = 1:length(sac_1)
    %add columns for time values instead of index
    sac_1t(i,8) = pupilR.pupil_timestamp(sac_1t(i,1));
    sac_1t(i,9) = pupilR.pupil_timestamp(sac_1t(i,2));
end

sac_2t = sac_2; %copies to retain original
for i = 1:length(sac_2)
    %add columns for time values instead of index
    sac_2t(i,8) = pupilL.pupil_timestamp(sac_2t(i,1));
    sac_2t(i,9) = pupilL.pupil_timestamp(sac_2t(i,2));
end

%% Heatmap for saccade end points

% binocular
sac_bi = binsacc(sac_2, sac_1); % looking for overlap in INDEX
sac_bit = binsaccT(sac_2t, sac_1t); % looking for overlap in TIME
% looking for time overlap seems to be working better since the Pupil Labs
% system doesn't guarentee that the L and R data lengths are the same

% bino-detected right eye saccades
r_saccade_start = pupilR(sac_bit(:,8),:);
r_saccade_end = pupilR(sac_bit(:,9),:);
r_angleX = r_saccade_end.angleX - r_saccade_start.angleX;
r_angleY = r_saccade_end.angleY - r_saccade_start.angleY;
r_angleXY = [r_angleX, r_angleY];

% bino-detected left eye saccades
l_saccade_start = pupilL(sac_bit(:,10),:);
l_saccade_end = pupilL(sac_bit(:,11),:);
l_angleX = l_saccade_end.angleX - l_saccade_start.angleX;
l_angleY = l_saccade_end.angleY - l_saccade_start.angleY;
l_angleXY = [l_angleX, l_angleY];

% difference
diff_angleX = r_angleX - l_angleX;
diff_angleY = r_angleY - l_angleY;
diff_angleXY = [diff_angleX, diff_angleY];

%% Saccade table

colNames = {'pupil_timestamp(s)','mono/bino','eye(r/l)'};

% bino saccades in right/left eye
bino_right = table(r_saccade_start.pupil_timestamp, repmat('bino',height(r_saccade_start),1),repmat('r',height(r_saccade_start),1),'VariableNames',colNames);
bino_left = table(l_saccade_start.pupil_timestamp, repmat('bino',height(l_saccade_start),1),repmat('l',height(l_saccade_start),1),'VariableNames',colNames);

% mono saccades in right/left eye
mono_right = table(sac_1t(:,8), repmat('mono',height(sac_1t(:,8)),1),repmat('r',height(sac_1t(:,8)),1),'VariableNames',colNames);
mono_left = table(sac_2t(:,8), repmat('mono',height(sac_2t(:,8)),1),repmat('l',height(sac_2t(:,8)),1),'VariableNames',colNames);

% combine into one big table
table = [bino_right; bino_left; mono_right; mono_left];
table = sortrows(table,{'pupil_timestamp(s)','mono/bino'}); % sort by time

time_idxs1 = zeros(size(sac_1(:,1)));

for i = 1:length(sac_1(:,1))
    time_idxs1(i) = pupilR.pupil_timestamp(sac_1(i,1));
end 

time_idxs2 = zeros(size(sac_2(:,1)));
for i = 1:length(sac_2(:,1))
    time_idxs2(i) = pupilL.pupil_timestamp(sac_2(i,1));
end 

time_idxs3 = zeros(size(sac_bit(:,8)));
for i = 1:length(sac_bit(:,8))
    time_idxs3(i) = pupilR.pupil_timestamp(sac_bit(i,8));
end 

time_idxs4 = zeros(size(sac_bit(:,10)));
for i = 1:length(sac_bit(:,10))
    time_idxs4(i) = pupilL.pupil_timestamp(sac_bit(i,10));
end 

time_idxs5 = zeros(size(sac_1(:,2)));
for i = 1:length(sac_1(:,2))
    time_idxs5(i) = pupilR.pupil_timestamp(sac_1(i,2));
end

time_idxs6 = zeros(size(sac_2(:,2)));
for i = 1:length(sac_2(:,2))
    time_idxs6(i) = pupilL.pupil_timestamp(sac_2(i,2));
end

time_idxs7 = zeros(size(sac_bit(:,9)));
for i = 1:length(sac_bit(:,9))
    time_idxs7(i) = pupilR.pupil_timestamp(sac_bit(i,9));
end 

time_idxs8 = zeros(size(sac_bit(:,11)));
for i = 1:length(sac_bit(:,11))
    time_idxs8(i) = pupilL.pupil_timestamp(sac_bit(i,11));
end 

writematrix([time_idxs1; time_idxs2; time_idxs3; time_idxs4; time_idxs5; time_idxs6; time_idxs7; time_idxs8],'C:\Users\Zachp\Downloads\same_itgd_2023_07_03_PL\exports_adult\000\matrix.txt');
disp(length(time_idxs1)+length(time_idxs2)+length(time_idxs3)+length(time_idxs4)+length(time_idxs5)+length(time_idxs6)+length(time_idxs7)+length(time_idxs8))
% keyboard
close all
axis tight manual
v = VideoWriter("C:\Users\Zachp\Downloads\same_itgd_2023_07_03_PL\exports_adult\000\plot_video.avi");
open(v);

figure
ax = axes('Position',[.1 .1 .8 .3]);
ay = axes('Position',[.1 .6 .8 .3]);

hold(ax,'on')
hold(ay,'on')

title(ax,'X PUPIL Saccades')
title(ay,'Y PUPIL Saccades')

xlabel(ax,'Time (seconds)')
xlabel(ay,'Time (seconds)')

ylabel(ax,'X Angle position (degrees)')
ylabel(ay,'Y Angle position (degrees)')

ylim(ax, [-90   90  ])
ylim(ay, [-90   90  ])
yticks(ax, [-90 -60 -30 0 30 60 90])
yticks(ay, [-90 -60 -30 0 30 60 90])
set(gcf, 'Position', [100 100 700 700])

% x points
scatter(ax,pupilR.pupil_timestamp, pupilR.angleX, 8, D_RED , "filled")
scatter(ax,pupilL.pupil_timestamp, pupilL.angleX, 8, D_BLUE , "filled")
% y points
scatter(ay,pupilR.pupil_timestamp, pupilR.angleY, 8, D_RED , "filled")
scatter(ay,pupilL.pupil_timestamp, pupilL.angleY, 8, D_BLUE , "filled")

% begin x
plot(ax,pupilR.pupil_timestamp, pupilR.angleX,'o','MarkerSize',8,'MarkerEdgeColor',GREEN, 'MarkerFaceColor' ,GREEN, 'Color',BLUE,'LineWidth',1, ...
    'MarkerIndices', sac_1(:,1) ) % mono saccades
plot(ax,pupilL.pupil_timestamp, pupilL.angleX,'o','MarkerSize',8,'MarkerEdgeColor',GREEN, 'MarkerFaceColor' ,GREEN, 'Color',RED, 'LineWidth',1, ...
    'MarkerIndices', sac_2(:,1) ) % mono saccades
plot(ax,pupilR.pupil_timestamp, pupilR.angleX,'s','MarkerSize',12,'MarkerEdgeColor',D_CYAN, 'Color',BLUE,'LineWidth',3, ...
    'MarkerIndices', sac_bit(:,8) ) % bino saccades
plot(ax,pupilL.pupil_timestamp, pupilL.angleX,'s','MarkerSize',12,'MarkerEdgeColor',D_CYAN, 'Color',RED, 'LineWidth',3, ...
    'MarkerIndices', sac_bit(:,10) )% bino saccades
% end x
plot(ax,pupilR.pupil_timestamp, pupilR.angleX,'o','MarkerSize',8,'MarkerEdgeColor',RED, 'MarkerFaceColor' ,RED, ...
    'MarkerIndices', sac_1(:,2) ) % mono saccades
plot(ax,pupilL.pupil_timestamp, pupilL.angleX,'o','MarkerSize',8,'MarkerEdgeColor',RED, 'MarkerFaceColor' ,RED, ...
    'MarkerIndices', sac_2(:,2) ) % mono saccades
plot(ax,pupilR.pupil_timestamp, pupilR.angleX,'s','MarkerSize',12,'MarkerEdgeColor',D_MAGENTA, 'Color',BLUE,'LineWidth',3, ...
    'MarkerIndices', sac_bit(:,9) )% bino saccades
plot(ax,pupilL.pupil_timestamp, pupilL.angleX,'s','MarkerSize',12,'MarkerEdgeColor',D_MAGENTA, 'Color',RED, 'LineWidth',3, ...
    'MarkerIndices', sac_bit(:,11) ) % bino saccades
% begin y
plot(ay,pupilR.pupil_timestamp, pupilR.angleY,'o','MarkerSize',8,'MarkerEdgeColor',GREEN, 'MarkerFaceColor' ,GREEN, 'Color',BLUE,'LineWidth',1, ...
    'MarkerIndices', sac_1(:,1) )
plot(ay,pupilL.pupil_timestamp, pupilL.angleY,'o','MarkerSize',8,'MarkerEdgeColor',GREEN, 'MarkerFaceColor' ,GREEN, 'Color',RED, 'LineWidth',1, ...
    'MarkerIndices', sac_2(:,1) )
plot(ay,pupilR.pupil_timestamp, pupilR.angleY,'s','MarkerSize',12,'MarkerEdgeColor',D_CYAN, 'Color',BLUE,'LineWidth',3, ...
    'MarkerIndices', sac_bit(:,8) )
plot(ay,pupilL.pupil_timestamp, pupilL.angleY,'s','MarkerSize',12,'MarkerEdgeColor',D_CYAN, 'Color',RED, 'LineWidth',3, ...
    'MarkerIndices', sac_bit(:,10) )
% end y
plot(ay,pupilR.pupil_timestamp, pupilR.angleY,'o','MarkerSize',8,'MarkerEdgeColor',RED, 'MarkerFaceColor' ,RED, ...
    'MarkerIndices', sac_1(:,2) )
plot(ay,pupilL.pupil_timestamp, pupilL.angleY,'o','MarkerSize',8,'MarkerEdgeColor',RED, 'MarkerFaceColor' ,RED, ...
    'MarkerIndices', sac_2(:,2) )
plot(ay,pupilR.pupil_timestamp, pupilR.angleY,'s','MarkerSize',12,'MarkerEdgeColor',D_MAGENTA, 'Color',BLUE,'LineWidth',3, ...
    'MarkerIndices', sac_bit(:,9) )
plot(ay,pupilL.pupil_timestamp, pupilL.angleY,'s','MarkerSize',12,'MarkerEdgeColor',D_MAGENTA, 'Color',RED, 'LineWidth',3, ...
    'MarkerIndices', sac_bit(:,11) )

% create lines that show the current time of the video on the graphs
for i = 0:vidObj.NumFrames-1
    if i/vidObj.FrameRate < 0.5
        xlim(ax, [0   i/vidObj.FrameRate+0.5  ])
        xlim(ay, [0   i/vidObj.FrameRate+0.5  ])
    elseif i/vidObj.FrameRate >= 0.5
        xlim(ax, [i/vidObj.FrameRate-0.5   i/vidObj.FrameRate+0.5  ])
        xlim(ay, [i/vidObj.FrameRate-0.5   i/vidObj.FrameRate+0.5  ])
    end
    
    xl_x = xline(ax,i/vidObj.FrameRate,'-');
    xl_y = xline(ay,i/vidObj.FrameRate,'-');

    frame = getframe(gcf);
    writeVideo(v,frame)
    delete(xl_y)
    delete(xl_x)

    hold(ax,'off')
    hold(ay,'off')
end

close(v)
