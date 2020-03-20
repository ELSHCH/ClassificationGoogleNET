function helperCreateRGBfromTF_Wavelet(ti,Data,parentFolder,childFolder)
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

for ii = 1:1
%     %cfs = abs(fb.wt(data(ii,:)));  
%     cfs=cwt(data(ii,:));
%     I = rescale(cfs);
%    % I=I/norm(I(:));
%     I=uint8(I*100);
%     im = ind2rgb(I,jet(128));
    [wt1,f1] = cwt(data(ii,:,1),Fs,'morse');
    fb1 = cwtfilterbank('SignalLength',signalLength,'SamplingFrequency',Fs,...
         'FrequencyLimits',[min(f1) min(f1)*10]);

    xrec1 = icwt(wt1,f1,[min(f1) max(f1)*0.1],'SignalMean',mean(data(ii,:,1)));
    [wt2,f2] = cwt(data(ii,:,2),Fs,'morse');
    fb2 = cwtfilterbank('SignalLength',signalLength,'SamplingFrequency',Fs,...
         'FrequencyLimits',[min(f2) min(f1)*10]);

    xrec2 = icwt(wt2,f2,[min(f2) max(f2)*0.1],'SignalMean',mean(data(ii,:,2)));

    [wcoh,~,f,coi] = wcoherence(xrec2,xrec1);
   % period = seconds(period);
% coi = seconds(coi);
% h = pcolor(t,log2(period),wcoh);
% h.EdgeColor = 'none';
% ax = gca;
% ytick=round(pow2(ax.YTick),3);
% ax.YTickLabel=ytick;
% ax.XLabel.String='Time';
% ax.YLabel.String='Period';
% ax.Title.String = 'Wavelet Coherence';
% hcol = colorbar;
% hcol.Label.String = 'Magnitude-Squared Coherence';
% hold on;
% plot(ax,t,log2(coi),'w--','linewidth',2);


    figure
    tms = (0:tr-1)/Fs;
    length(f)
    size(wcoh)
    length(tms)
    period=1./f;
    fig=pcolor(tms,period,wcoh);
    fig.EdgeColor = 'none';
    ax = gca;
ytick=round(pow2(ax.YTick),3);
ax.YTickLabel=ytick;
ax.XLabel.String='Time';
ax.YLabel.String='Period';
ax.Title.String = 'Wavelet Coherence';
hcol = colorbar;
hcol.Label.String = 'Magnitude-Squared Coherence';
hold on;
plot(ax,tms,log2(coi),'w--','linewidth',2);
    
    shading flat
%     figure
%      cwt(xrec,'FilterBank',fb);
%     [cfs,frq]=cwt(xrec,'FilterBank',fb);
%     tms = (0:r-1)/Fs;
%     fig=surface(tms,frq,abs(cfs));
%     shading flat
  
    
    saveas(fig,'Myplot.jpg');
    im=imread('Myplot.jpg');
    imshow(im)
   % im = ind2rgb(rescale(cfs),jet(128));
    imgLoc = fullfile(imageRoot,char(labels(ii)));
    imFileName = strcat(char(labels(ii)),'_',num2str(ii),'.jpg')
    imwrite(imresize(im,[224 224]),fullfile(imgLoc,imFileName));
end
end

