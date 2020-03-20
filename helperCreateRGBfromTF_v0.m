function helperCreateRGBfromTF_v0(ti,Data,parentFolder,childFolder)
% This function is only intended to support the ECGAndDeepLearningExample.
% It may change or be removed in a future release.

imageRoot = fullfile(parentFolder,childFolder);

data = Data.Data;
labels = Data.Labels;
delT = ti(3)-ti(2);
Fs = 1/delT;

[~,signalLength ] = size(data)

fb = cwtfilterbank('SignalLength',signalLength,'VoicesPerOctave',12);
r = size(data,1);
tr=size(data,2);

for ii = 1:r
%     %cfs = abs(fb.wt(data(ii,:)));  
%     cfs=cwt(data(ii,:));
%     I = rescale(cfs);
%    % I=I/norm(I(:));
%     I=uint8(I*100);
%     im = ind2rgb(I,jet(128));
    figure
    cwt(data(ii,:),Fs);
    [cfs,frq]=cwt(data(ii,:),Fs);
    %cfs=normalize(cfs,'norm',1);
    tms = (0:tr-1)/Fs;
    fig=surface(tms,frq,abs(cfs));
    shading flat;
  
    
    saveas(fig,'Myplot.jpg');
    im=imread('Myplot.jpg');
   % close 
   % imshow(im)
   % im = ind2rgb(rescale(cfs),jet(128));
    imgLoc = fullfile(imageRoot,char(labels(ii)));
    imFileName = strcat(char(labels(ii)),'_',num2str(ii),'.jpg');
    imwrite(imresize(im,[224 224]),fullfile(imgLoc,imFileName));
end
end

