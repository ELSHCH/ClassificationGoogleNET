%load NormalizedBoknis

% After downloading the data from GitHub, unzip the file in your temporary directory.
% tempdir = 'C:\Users\eshchekinova\Documents\BoknisData';
% load(fullfile(tempdir,'BoknisDataSpectro.mat'))
% parentDir = tempdir;
% dataDir = 'SpectroBoknisData';
% helperCreateDirectories(BoknisDataSpectro,parentDir,dataDir
currdir='C:\Users\eshchekinova\Documents\BoknisData\GoogleClass';
tempdir = 'C:\Users\eshchekinova\Documents\BoknisData\GoogleClass\SpectroBoknisData';

% load(fullfile(tempdir,'DataBoknis.mat'))
% parentDir = tempdir;
% dataDir = 'FiguresDataBoknis';
% helperCreateDirectories(DataBoknis,parentDir,dataDir)

load(fullfile(tempdir,'BoknisDataSpectro.mat'))
parentDir = tempdir;
dataDir = 'SpectroBoknisData';
helperCreateDirectories(BoknisDataSpectro,parentDir,dataDir)

DataM=BoknisDataSpectro;

%helperPlotReps(BoknisDataSpectro)
%helperCreateRGBfromTF_v0(new_time_sec,BoknisDataSpectro,parentDir,dataDir)
%helperCreateRGBfromTF_Spectro(new_time_sec,BoknisDataSpectro,parentDir,dataDir)
%helperCreateRGBfromTF(new_time_sec,BoknisDataSpectro,parentDir,dataDir)
%helperCreateRGBfromTF_Spectro(new_time_sec,BoknisDataSpectro,parentDir,dataDir)
cd(currdir);


%clear all
% %% Plot the sequence and calculated scores
%addpath('C:\Users\eshchekinova\Documents\BoknisData\SpectroBoknisData_v1');
addpath('C:\Users\eshchekinova\Documents\BoknisData\GoogleClass\SpectroBoknisData');
addpath('C:\Users\eshchekinova\Documents\BoknisData\GoogleClass\Networks');
dirFigures = 'C:\Users\eshchekinova\Documents\BoknisData\GoogleClass\SpectroBoknisData\SpectroBoknisData';
%addpath('C:\Users\eshchekinova\Documents\BoknisData\FiguresBoknisData');
%load DataBoknis
load UpdateBoknis
%load NormalizedBoknis19
%load BoknisDataSpectro
%load DataBoknis
%load AlexGoogleNetsTrainedFullSeries 
%load AlexGoogleNetsTrainedFullSeries
%load AlexGoogleNetsTrainedGMDistribution
%load GoogleNetGMDistribution 
%load AlexGoogleNetsTrained_v1 
%load AlexGoogleNetsTrained_GMM
load AlexGoogleNetsTrained_Wavelet
%googlenetGM;
% Xstart_sec=[1531193880,1531899535,1532287813,1532671269,1533343545,1534734210,...
%             1535641577,1536713419,1538482330,1538869547,1539837495,1557543679,1558504873,...
%             1559284228,1559823960,1560490931,1561024681,1561561230,1563203673,1564354135,1565961078];
%   
% Xend_sec=[1531348516,1532051523,1532417465,1532991442,1534458605,1535009719,...
%           1536058699,1536998189,1538567220,1538990324,1540326774,1558165310,1559193550,...
%           1559514591,1560329156,1560680585,1561428009,1562301130,1563547480,1565745861,1566000000];

Xstart_sec=[1531194323,1531910051, 1532289700, 1532673600,1532868800,1533355321, 1533748519, 1534055108, 1534247191, ...
    1534725139, 1535644639, 1535953601,1536693010,...
    1537964510, 1538481513, 1538855849, 1539845487, 1540051650, 1540192989, 1540271894, 1541235209];
Xend_sec=[1531340179,1532160000, 1532501904,1532838000,1532986200,1533620464,1533915629, 1534211856, 1534481791, ...
    1535012599, 1535844692,1536061807, 1536976997,...
     1538245328, 1538545275, 1538975136, 1540022692, 1540173860, 1540267378, 1540324498, 1541510810];

nClasses=2;
net = trainedGN;
%net = googlenetGM
%trainedGN=googlenetGM;
netAl = trainedAN
%netAl=googlenetGM;
%trainedAN=googlenetGM;
classNamesGN = net.Layers(end).ClassNames
classNamesAG = netAl.Layers(end).ClassNames;
inputSizeGN =net.Layers(1).InputSize;
inputSizeAN =netAl.Layers(1).InputSize;
%DataM=DataBoknis;
DataM=BoknisDataSpectro;
nImages=length(DataM.Data(:,1,1));
seqScores=zeros(nImages,2);
%tempdir = 'C:\Users\eshchekinova\Documents\BoknisData\SpectroBoknisData_v1';
%tempdir = 'C:\Users\eshchekinova\Documents\GoogleClass\SpectroBoknisData';
%tempdir = 'C:\Users\eshchekinova\Documents\BoknisData\FiguresDataBoknis';
% imgClassNameGN=imread(strcat(tempdir,'\YEVENT\','YEVENT_38.jpg'));
% imgClassNameAN=imread(strcat(tempdir,'\YEVENT\','YEVENT_38.jpg'));
%folderLabels = unique(BoknisDataSpectro.Labels)
folderLabels = unique(DataM.Labels)
eventType = folderLabels
indNevent = find(ismember(DataM.Labels,eventType(2)));
indYevent = find(ismember(DataM.Labels,eventType(1)));
ind=[1:1:length(DataM.Data(:,1,1))];
length(ind)
k1=1;
k2=1;
label_Y_GN=zeros(length(DataM.Data(:,1,1)),1);
label_Y_AN=zeros(length(DataM.Data(:,1,1)),1);
for i=1:length(DataM.Data(:,1,1))
    if (indNevent(k1)==ind(i))&&(k1<length(indNevent))
      imgClassNameGN=imread(strcat(dirFigures,'\NEVENT\','NEVENT_',num2str(i),'.jpg'));
      imgClassNameAN=imread(strcat(dirFigures,'\NEVENT\','NEVENT_',num2str(i),'.jpg'));
      k1=k1+1;
    elseif (indYevent(k2)==ind(i))&&(k2<length(indYevent))
      imgClassNameGN=imread(strcat(dirFigures,'\EVENT\','EVENT_',num2str(i),'.jpg'));
      imgClassNameAN=imread(strcat(dirFigures,'\EVENT\','EVENT_',num2str(i),'.jpg'));
      k2=k2+1;
    end;  
% figure
% imshow(imgClassName);

imgClassNameGN=imresize(imgClassNameGN,inputSizeGN(1:2));
imgClassNameAN=imresize(imgClassNameAN,inputSizeAN(1:2));
% figure
% imshow(imgClassName)
[labelGN,scoresGN] = classify(trainedGN,imgClassNameGN);
[labelAN,scoresAN] = classify(trainedAN,imgClassNameAN);
if labelGN=='EVENT'
       label_Y_GN(i)=1;
end; 
if labelAN=='EVENT'
       label_Y_AN(i)=1;
end;  
for j=1:2
seqScoresAN(i,j)=scoresAN(j);
seqScoresGN(i,j)=scoresGN(j);
% if labelGN(j)=='EVENT'
%       label_Y_GN(i,j)=1;
% end; 
% if labelAN(j)=='EVENT'
%       label_Y_AN(i,j)=1;
%   end;  
end;
end;

tt=linspace(new_time_sec(1),new_time_sec(end),length(DataM.Data(:,1,1)));
%xx=interp1(new_time_sec,nO2,tt);

seqProbAN = zeros(length(new_time_sec),2);
seqProbGN = zeros(length(new_time_sec),2);
seq_Y_GN = zeros(length(new_time_sec),2);
seq_Y_AN = zeros(length(new_time_sec),2);
length(tt)
length(seqScoresAN(:,1))
size(DataM.Data)
for i=1:2
seqProbAN(:,i)=interp1(tt,seqScoresAN(:,i),new_time_sec);
seqProbGN(:,i)=interp1(tt,seqScoresGN(:,i),new_time_sec);
end; 
seq_Y_AN=interp1(tt,label_Y_AN,new_time_sec);
 seq_Y_GN=interp1(tt,label_Y_GN,new_time_sec);
 'X'
 size(X)
xOxy=interp1(new_time_sec,X(:,1),tt);
%xPress=interp1(new_time_sec,X(:,1),tt);
figure;
% subplot(2,1,1);
% %X(1,4)=NaN;
% seqProbAN1(1)=NaN;
% %X(end,4)=NaN;
% seqProbAN1(end)=NaN;
% patch(new_time_sec,X(:,1),seqProbAN(:,1)*100,'EdgeColor','interp','Marker','.','MarkerFaceColor','flat');
% title('Oxygen');
% xlim([new_time_sec(1) new_time_sec(end)]);
% ylim([-0.01 0.02]);
subplot(2,1,1);
plot(new_time_sec,X(:,1),'LineWidth',1,'Color','r');
num_events=21;
 for i=1:num_events
 xx = [Xstart_sec(i) Xend_sec(i) Xend_sec(i) Xstart_sec(i)];

 yy = [-1 -1 1 1];
 patch(xx,yy,[0.3 0. 0.8])
 alpha(.1)
 end;
%title('Temperature');
xlim([new_time_sec(1) new_time_sec(end)]);
ylim([-1 1]);
subplot(2,1,2);
%plot(tt,seqScoresGN(:,1)*100,'r','LineWidth',3);
%hold on;

plot(new_time_sec,seqProbAN(:,1)*100,'Color',[0.2 0.8 0.2],'LineWidth',2);
hold on
plot(new_time_sec,seqProbGN(:,1)*100,'LineWidth',2,'Color',[0.8 0.1 0.1]);
legend({'Probability Score (AlexNet) %','Probability Score (GoogleNet) %'})
ylim([0 100]);
xlim([new_time_sec(1) new_time_sec(end)]);
cd(currdir);
savefig('Classify.fig');
%title('AlexNet, 142 layers');
title('Probability Scores');