function calcPlotProbEVENT
%-------------------------------------------------------------------------------
%  Predict probability of sudden change of oxygen based on LSTM prediction and 
%  pretrained GoogleNet model
%
%
%
%
%
%  Last modified E.Shchekinova 18.03.2020 
%-------------------------------------------------------------------------------
clear all
dirCurrent ='C:/Users/eshchekinova/Documents/BoknisData/LSTMGoogleClass';
dirLSTMModel='C:/Users/eshchekinova/Documents/BoknisData/LSTMPred/Model';
dirData='C:/Users/eshchekinova/Documents/BoknisData/LSTMPred/PredictionData';
fileName='InPrediction.dat';
cd(dirData);         
load('NormalizedBoknis.mat');
categories_original=categories;
n_Var_f=size(X,2);
for si=1:n_Var_f
% Standartize data

 mu(si) = mean(X(:,si));
 sig(si) = std(X(:,si));

 X(:,si)=(X(:,si)-mu(si))/sig(si);
 max_X_f=max(X(:,si));
 X(:,si)=X(:,si)/max_X_f;
end;
cd(dirLSTMModel);
[Algorithm_Scheme,choice_training,time_start,P_horizon_s,n_points,nVar,sampleSize,fileData]=...
                  InputParam(fileName);
P_horizon_s=P_horizon_s/3600;       
[n_Var,categories_final,X_predictors]=selectCategories('ListCategories.dat',categories,X,nVar);
fileData = strcat(fileData,'_',Algorithm_Scheme,'_',choice_training,'_',num2str(sampleSize),'_',num2str(P_horizon_s),'_', ...
             num2str(n_points),'_',num2str(n_Var),'.mat'); 
cd(dirData);            
load(fileData);
categories_predicted=categories;
hold on
cd(dirData);     
s1=1;
nVar=size(X_f,2);
nOb=size(X_f,1);
plot(t_f,X_f)
ind=find(new_time_sec>t_f(1));
ind(1)
t_f(1)
%for i=1:length(new_time_sec)
%    for k=1:length(t_f)
%     ind=find(new_time_sec<t_f)
k=0;
for i=1:length(categories_original)
 for j=1:length(categories_predicted)
      if strcmp(cellstr(categories_original{i}),cellstr(categories_predicted{j}))==1
       k=k+1;   
      end;    
 end;
end;
X_s=zeros(ind(1)+nOb-1,k);
size(X_s)
k=0;
for i=1:length(categories_original)
 for j=1:length(categories_predicted)
      if strcmp(cellstr(categories_original{i}),cellstr(categories_predicted{j}))==1
       k=k+1;   
       X_s(1:ind(1),k)=X(1:ind(1),i); % calculate RMSE for available prediction
       X_s(ind(1)+1:ind+nOb-1,k)=X_f(2:nOb,j); % create a copy of original time series 
      end;    
 end;
end;
%X_s(1:ind,1:nVar)=X(1:ind,1:nVar); % calculate RMSE for available prediction
t_s(1:ind(1))=new_time_sec(1:ind(1));
%        t_s(s1)=new_time_sec(i);
%        s1=s1+1
%      else
%X_s(ind+1:ind+nOb-1,1:nVar)=X_f(2:nOb,1:nVar); % create a copy of original time series 
t_s(ind(1)+1:ind(1)+nOb-1)=t_f(2:nOb);
%        t_s(s1)=t_f(k); % create a copy of time series
%        s1=s1+1;
%      end;
%    end;
%  end;
 del_t=min(t_f(2)-t_f(1),new_time_sec(2)-new_time_sec(1));
 num=floor((t_s(end)-t_s(1))/del_t);
 t_int=linspace(t_s(1),t_s(end),num);
 for k=1:nVar
  x_int(:,k)=interp1(t_s,X_s(:,k),t_int);
 end;
 plot(t_s(:),X_s(:,1),'k');
%  hold on
%  plot(t_int,x_int(:,1),'o');
X=X_s;
new_time_sec=t_s;
dirGoogleNet='C:/Users/eshchekinova/Documents/BoknisData/GoogleClass';
cd(dirGoogleNet);
save UpdateBoknis X new_time_sec
cd(dirCurrent);
%CreateBoknisData;