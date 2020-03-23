function helperCreateRGBfromTF_Spectro(ti,Data,parentFolder,childFolder)
% This function is only intended to support the ECGAndDeepLearningExample.
% It may change or be removed in a future release.

imageRoot = fullfile(parentFolder,childFolder);

data = Data.Data;
labels = Data.Labels;

[~,signalLength] = size(data);

%fb = cwtfilterbank('SignalLength',signalLength,'VoicesPerOctave',12);

delT=ti(2)-ti(1);
Fs=1/delT;
% [wt,f] = cwt(data,Fs,'morse');
% fb = cwtfilterbank('SignalLength',numel(ti),'SamplingFrequency',Fs,...
%    'FrequencyLimits',[min(f) max(f)]);
% 
% xrec = icwt(wt,f,[min(f) min(f)*1.5],'SignalMean',mean(data));
% % fb = cwtfilterbank('SignalLength',numel(cut_time_hour(169:end-168)),'SamplingFrequency',Fs,...
% %    'FrequencyLimits',[0 10]);
% cwt(xrec,'FilterBank',fb);

r = size(data,1);
for ii = 1:r
%    % cfs = abs(fb.wt(data(ii,:)));
%     cfs = abs(fb.wt(xrec));
%     I = rescale(cfs);
%    % I=I/norm(I(:));
%     I=uint8(I*100);
%     im = ind2rgb(I,jet(128));
%    % im = ind2rgb(rescale(cfs),jet(128));
    [wt,f] = cwt(data(ii,:),Fs,'morse');
    fb = cwtfilterbank('SignalLength',signalLength,'SamplingFrequency',Fs,...
         'FrequencyLimits',[min(f) max(f)]);

    xrec = icwt(wt,f,[min(f) max(f)*0.1],'SignalMean',mean(data(ii,:)));

    figure
     cwt(xrec,'FilterBank',fb);
    [cfs,frq]=cwt(xrec,'FilterBank',fb);
    tms = (0:r-1)/Fs;
    fig=surface(tms,frq,abs(cfs));
    shading flat;
  
    
    saveas(fig,'Myplot.jpg');
    im=imread('Myplot.jpg');
    imshow(im)

    imgLoc = fullfile(imageRoot,char(labels(ii)));
    imFileName = strcat(char(labels(ii)),'_',num2str(ii),'.jpg')
    imwrite(imresize(im,[224 224]),fullfile(imgLoc,imFileName));
end
end
