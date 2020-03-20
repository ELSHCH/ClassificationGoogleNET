function helperCreateRGBfromTF(Data_sum,Data,parentFolder,childFolder)
% This function is only intended to support the ECGAndDeepLearningExample.
% It may change or be removed in a future release.

imageRoot = fullfile(parentFolder,childFolder);

data = Data.Data;
labels = Data.Labels;

%[Mx,vertDim] = MakeMatrix(data);
Nwindows = length(data(:,1,1));
Lwindows= length(data(1,:,1));
Nscale= length(data(1,1,:));
%Nscale=8;
dd=zeros(Lwindows,Nscale);
tt=linspace(0,Lwindows-1,Lwindows); 
y=linspace(0,Nscale-1,Nscale);
%y=linspace(1,8,8);

for ii = 1:Nwindows
    ii
    %fig=plot(1:Lwindows,data(ii,1:Lwindows));
    for i=1:Lwindows
    for j=1:Nscale
        dd(i,j)=data(ii,i,j);
    end;
    end;
    %fig=pcolor(tt,y,dd(1:Lwindows,1:Nscale)');
    %fig=plot(tt,dd(1:Lwindows,8),'.','MarkerSize',24);
    fig=scatter(dd(1:Lwindows,1),dd(1:Lwindows,2),80,'filled','o');
    %fig= histogram2(dd(1:Lwindows,8),dd(1:Lwindows,1),'DisplayStyle','tile','ShowEmptyBins','on');
    hold on
    scatter(dd(1:Lwindows,3),dd(1:Lwindows,2),80,'MarkerFaceColor',[.6 .1 .7],'MarkerEdgeColor',[.6 .1 .7]);
%     scatter(dd(1:Lwindows,3),dd(1:Lwindows,3),80,'MarkerFaceColor',[0 .7 .7],'MarkerEdgeColor',[0 .7 .7]);
%     scatter(dd(1:Lwindows,2),dd(1:Lwindows,5),60,'MarkerFaceColor',[.6 .1 .7],'MarkerEdgeColor',[.6 .1 .7]);
%     scatter(dd(1:Lwindows,5),dd(1:Lwindows,7),60,'MarkerFaceColor',[.1 .1 .9],'MarkerEdgeColor',[.1 .1 .9]);
    ylim([-1.,1.]);
    xlim([-1.,1.]);
    saveas(fig,'Myplot.jpg');
    im=imread('Myplot.jpg');
    imshow(im)
%     I = rescale(Mx);
%     I=I/norm(I(:));
%     I=uint8(I*100);
%      im = ind2rgb(I,jet(128));
    % im = ind2rgb(rescale(cfs),jet(128));
    imgLoc = fullfile(imageRoot,char(labels(ii)));
    imFileName = strcat(char(labels(ii)),'_',num2str(ii),'.jpg')
    imwrite(imresize(im,[224 224]),fullfile(imgLoc,imFileName));
end

