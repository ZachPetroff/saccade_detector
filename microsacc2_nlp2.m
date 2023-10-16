function [sac, radius] = microsacc2_nlp2(x,vel,VFAC,MINDUR)
%-------------------------------------------------------------------
%
%  FUNCTION microsacc.m
%  Detection of monocular candidates for microsaccades;
%  Please cite: Engbert, R., & Mergenthaler, K. (2006) Microsaccades 
%  are triggered by low retinal image slip. Proceedings of the National 
%  Academy of Sciences of the United States of America, 103: 7192-7197.
%
%
%  (Version 2.1, 03 OCT 05)
%
% NLP Modification - Returns empty and gives warnings in the beginning
%                    checking section.  Used to give errors.  Done
%                    for trials that are blank.
%
%-------------------------------------------------------------------
%
%  INPUT:
%
%  x(:,1:2)         position vector
%  vel(:,1:2)       velocity vector
%  VFAC             relative velocity threshold
%  MINDUR           minimal saccade duration
%
%  OUTPUT:
%
%  sac(1:num,1)   onset of saccade
%  sac(1:num,2)   end of saccade
%  sac(1:num,3)   peak velocity of saccade (vpeak)
%  sac(1:num,4)   horizontal component     (dx)
%  sac(1:num,5)   vertical component       (dy)
%  sac(1:num,6)   horizontal amplitude     (dX)
%  sac(1:num,7)   vertical amplitude       (dY)
%
%---------------------------------------------------------------------


% NLP - Aug 2013
DEBUG = 0;

sac    = [];
radius = [];
% compute threshold
msdx = sqrt( median(vel(:,1).^2,"omitnan") - (median(vel(:,1), "omitnan"))^2);
msdy = sqrt( median(vel(:,2).^2,"omitnan") - (median(vel(:,2), "omitnan"))^2);
if msdx<realmin
    % possibly an error here
    msdx = sqrt( mean(vel(:,1).^2,"omitnan") - (mean(vel(:,1),"omitnan"))^2 );
    if msdx<realmin
        warning('msdx<realmin in microsacc.m');
        return
    end
end
if msdy<realmin
    msdy = sqrt( mean(vel(:,2).^2, "omitnan") - (mean(vel(:,2), "omitnan"))^2 );
    if msdy<realmin
        warning('msdy<realmin in microsacc.m');
        return
    end
end
radiusx = VFAC*msdx;
radiusy = VFAC*msdy;
radius = [radiusx radiusy];

% compute test criterion: ellipse equation
test = (vel(:,1)/radiusx).^2 + (vel(:,2)/radiusy).^2;
indx = find(test>1); % find values where the x AND y velocities were greater than the radius/threshold

% determine saccades
N = length(indx); 
tmp_sac = [];
nsac = 0;
dur = 1;
a = 1;
k = 1;
while k<N
    if indx(k+1)-indx(k)==1
        dur = dur + 1;
    else
        if dur>=MINDUR
            nsac = nsac + 1;
            b = k;
            tmp_sac(nsac,:) = [indx(a) indx(b)]; % mark the beginning and end of the given saccade
        end
        a = k+1;
        dur = 1;
    end
    k = k + 1;
end




% check for minimum duration
if dur>=MINDUR
    nsac = nsac + 1;
    b = k;
    tmp_sac(nsac,:) = [indx(a) indx(b)];
end

% At this point 'tmp_sac' has a list of starting and ending points for saccades
% Now we need to check whether the eye was at rest before the
% saccade.  This is done by check whether the mean x & y eye position fit
% within a predefined window


KS  = 20; % buffer window before and after a saccade
WIN = 15; % window of noise that is allowed in order to count a saccade as "good" (in units of pixels?)
nsac2 = 0;
for ii = 1:size(tmp_sac,1)
   
   sac_good(ii) = 0;
   
   if tmp_sac(ii,1) > 20 % make sure we have a baseline before the first saccade
      % Indexes
      s1 = tmp_sac(ii,1)-KS;
      s2 = tmp_sac(ii,2)+KS;
      
      %s11 = sac(ii,1);
      %s22 = sac(ii,2);
      
      %
      %xm = x(s11,1); %X Mean
      %ym = x(s11,2); %Y Mean
      xm =mean( x(s1:s1+KS, 1)); %X Mean
      ym =mean( x(s1:s1+KS, 2)); %Y Mean
      
      x_data = x(s1:s1+KS,1);
      y_data = x(s1:s1+KS,2);
      
      x_diff = x_data - xm > WIN;
      y_diff = y_data - ym > WIN;
      
      if max(x_diff) ~=  1 && max(y_diff) ~= 1
         sac_good(ii) = 1;
      end
      
      if DEBUG
         figure
         subplot(1,2,1);
         hold on
         plot(x(s1:s2,1),'r-');
         plot(x(s1:s2,2),'b-');
         
         
         
         plot([1 KS], [xm     xm    ], 'r--','LineWidth',2);
         plot([1 KS], [xm+WIN xm+WIN], 'k--','LineWidth',1);
         plot([1 KS], [xm-WIN xm-WIN], 'k:', 'LineWidth',1);
         
         plot([1 KS], [ym     ym    ], 'b--','LineWidth',2);
         plot([1 KS], [ym+WIN ym+WIN], 'k--','LineWidth',1);
         plot([1 KS], [ym-WIN ym-WIN], 'k:', 'LineWidth',1);
         title(['Sac Good = ' num2str(sac_good(ii)) ]);
         
         subplot(1,2,2);
         hold on
         plot(vel(s1:s2,1),'r-');
         plot(vel(s1:s2,2),'b-');
         %pause(2);
         %keyboard;
      end %DEBUG
      
   end %sac(ii,1) > 20
   
   
   
   if sac_good(ii) == 1
      nsac2 = nsac2 + 1;
      sac(nsac2,:) = tmp_sac(ii,:);
   else
      
      %keyboard;
   end
   
end



% compute peak velocity, horizonal and vertical components
for s=1:nsac2
    % onset and offset
    a = sac(s,1); 
    b = sac(s,2); 
    % saccade peak velocity (vpeak)
    vpeak = max( sqrt( vel(a:b,1).^2 + vel(a:b,2).^2 ) );
    sac(s,3) = vpeak;
    % saccade vector (dx,dy)
    dx = x(b,1)-x(a,1); 
    dy = x(b,2)-x(a,2); 
    sac(s,4) = dx;
    sac(s,5) = dy;
    % saccade amplitude (dX,dY)
    i = sac(s,1):sac(s,2);
    [minx, ix1] = min(x(i,1));
    [maxx, ix2] = max(x(i,1));
    [miny, iy1] = min(x(i,2));
    [maxy, iy2] = max(x(i,2));
    dX = sign(ix2-ix1)*(maxx-minx);
    dY = sign(iy2-iy1)*(maxy-miny);
    sac(s,6:7) = [dX dY];
    
    %[th,r] = cart2pol(x(i,1),x(i,2));
    %sac(s,8) = max(r)-min(r);
end


%keyboard;




