clear all
%load DataInterp;
%load DataBoknisEck2018_Press;
dirGoogleNet='C:/Users/eshchekinova/Documents/BoknisData/GoogleClass';
load('UpdateBoknis.mat');

Xstart_sec=[1531194323,1531910051, 1532289700, 1532673600,1532868800,1533355321, 1533748519, 1534055108, 1534247191, ...
    1534725139, 1535644639, 1535953601,1536693010,...
    1537964510, 1538481513, 1538855849, 1539845487, 1540051650, 1540192989, 1540271894, 1541235209];
Xend_sec=[1531340179,1532160000, 1532501904,1532838000,1532986200,1533620464,1533915629, 1534211856, 1534481791, ...
    1535012599, 1535844692,1536061807, 1536976997,...
     1538245328, 1538545275, 1538975136, 1540022692, 1540173860, 1540267378, 1540324498, 1541510810];

 Nd=length(X(1,:));
 Ndata=length(X(:,1))
%xf(1:Ns,1:Nd)=scoreX(1:Ns,1:Nd);
%[diffX]=FilterTimeSeries(xf,Ns,Nd,numDiv);
delT=new_time_sec(2)-new_time_sec(1);
%Xf = filterX(X);
%Nd=length(diffX(1,:));
%diffX(1:Ns,1)=xf(1:Ns,1);
xf=X;
Lw=300;
Overlapw=10;
j=1;
while (j*(Lw-Overlapw)<=Ndata)
        j=j+1;
end;        
Nw=j-1
% Xstart_rec=[178];
% Xend_rec=[196];
% Xstart_prerec=[165];
x_event=[];
x_noevent=[];
k2=1;
k3=1;
Xwin=zeros(Lw,Nd,Nw);
for ii = 1:Nw
    Xwin(1:Lw,1:Nd,ii)=xf(1+(ii-1)*(Lw-Overlapw):Lw+(ii-1)*(Lw-Overlapw),1:Nd);
    %Xwin(1:Lw,ii)=Xf(1+(ii-1)*(Lw-Overlapw):Lw+(ii-1)*(Lw-Overlapw),1);
end;  
X_win=zeros(Lw,Nw,Nd);
 subplot(2,1,1);
%plot(new_time_sec,X(:,4),'r')
hold on
Y=zeros(Ndata,1);
for k=1:length(Xend_sec)
for i=1:Ndata
if ((new_time_sec(i)>=Xstart_sec(k))&&(new_time_sec(i)<=Xend_sec(k)))    
     Y(i)=1;
end;
end;
end;
for ii=1:Nw
   BoknisDataSpectro.Labels(ii)={'NEVENT'}';
end;
for k=1:length(Xend_sec)
for ii=1:Nw
       if numel(intersect(floor(new_time_sec(1+(ii-1)*(Lw-Overlapw))):floor(new_time_sec(Lw+(ii-1)*(Lw-Overlapw))),Xstart_sec(k):Xend_sec(k)))>20      
          BoknisDataSpectro.Labels(ii)={'EVENT'}';
         
          plot(new_time_sec(1+(ii-1)*(Lw-Overlapw):Lw+(ii-1)*(Lw-Overlapw)),X(1+(ii-1)*(Lw-Overlapw):Lw+(ii-1)*(Lw-Overlapw),1),'.')
        else  
        % plot(new_time_sec(1+(ii-1)*(Lw-Overlapw):Lw+(ii-1)*(Lw-Overlapw)),X(1+(ii-1)*(Lw-Overlapw):Lw+(ii-1)*(Lw-Overlapw),4),'.')
        end;
end;
end;
%subplot(2,1,2);
plot(new_time_sec,Y,'b')
%ylim([-0.01,0.02]);
X1=zeros(Nw,Lw,Nd);
Nd
for j=1:Nd
for ii=1:Nw
        X1(ii,1:Lw,1)=Xwin(1:Lw,j,ii);
        X1(ii,1:Lw,2)=Xwin(1:Lw,j,ii);
end;
end;
BoknisDataSpectro.Data=X1;

% subplot(2,1,1);
% plot(1:length(X),X(:,8))
% 
% subplot(2,1,2);
% hold on
% for i=1:k3-1
% plot(new_time_sec(1+(i-1)*(Lw-Overlapw)+(i-1)*Lw:(i-1)*(Lw-Overlapw)+i*Lw),X1(i,1:Lw,8));
% end;
dirSpectro='C:/Users/eshchekinova/Documents/BoknisData/GoogleClass/SpectroBoknisData';
cd(dirSpectro);
save BoknisDataSpectro BoknisDataSpectro new_time_sec Y