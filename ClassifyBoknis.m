%% Classify Time Series Using Wavelet Analysis and Deep Learning
% This example shows how to classify human electrocardiogram (ECG) signals
% using the continuous wavelet transform (CWT) and a deep convolutional
% neural network (CNN).
%
% Training a deep CNN from scratch is computationally expensive and
% requires a large amount of training data. In various applications, a
% sufficient amount of training data is not available, and synthesizing new
% realistic training examples is not feasible. In these cases, leveraging
% existing neural networks that have been trained on large data sets for
% conceptually similar tasks is desirable. This leveraging of existing
% neural networks is called transfer learning. In this example we adapt two
% deep CNNs, GoogLeNet and AlexNet, pretrained for image recognition to
% classify ECG waveforms based on a time-frequency representation.
%
% GoogLeNet and AlexNet are deep CNNs originally designed to classify
% images in 1000 categories. We reuse the network architecture of the CNN
% to classify ECG signals based on images from the CWT of the time series
% data. To run this example you must have Wavelet Toolbox(TM), Image
% Processing Toolbox(TM), Deep Learning Toolbox(TM), Deep Learning
% Toolbox(TM) Model _for GoogLeNet Network_ support package, and Deep
% Learning Toolbox(TM) Model _for AlexNet Network_ support package. To find
% and install the support packages use the MATLAB(TM) Add-On Explorer. The
% option within this example is set so that the training processes run on a
% CPU. If your machine has a GPU and Parallel Computing Toolbox(TM), you
% can accelerate the training processes by setting the option to run on the
% GPU. The data used in this example are publicly available from
% <https://physionet.org PhysioNet>.
%% Data Description
% In this example, you use ECG data obtained from three groups of people:
% persons with cardiac arrhythmia (ARR), persons with congestive heart
% failure (CHF), and persons with normal sinus rhythms (NSR). In total you
% use 162 ECG recordings from three PhysioNet databases:
% <https://www.physionet.org/physiobank/database/mitdb/ MIT-BIH Arrhythmia
% Database> [3][7], <https://www.physionet.org/physiobank/database/nsrdb/
% MIT-BIH Normal Sinus Rhythm Database> [3], and
% <https://www.physionet.org/physiobank/database/chfdb/ The BIDMC
% Congestive Heart Failure Database> [1][3]. More specifically, 96
% recordings from persons with arrhythmia, 30 recordings from persons with
% congestive heart failure, and 36 recordings from persons with normal
% sinus rhythms. The goal is to train a classifier to distinguish between
% ARR, CHF, and NSR.
%% Download Data
% The first step is to download the data from the
% <https://github.com/mathworks/physionet_ECG_data/ GitHub repository>. To
% download the data from the website, click |Clone or download| and select
% |Download ZIP|. Save the file |physionet_ECG_data-master.zip| in a folder
% where you have write permission. The instructions for this example assume
% you have downloaded the file to your temporary directory, |tempdir|, in
% MATLAB. Modify the subsequent instructions for unzipping and loading the
% data if you choose to download the data in folder different from
% |tempdir|. If you are familiar with Git, you can download the latest
% version of the tools (<https://git-scm.com/ git>) and obtain the data
% from a system command prompt using |git clone
% https://github.com/mathworks/physionet_ECG_data/|.
clear all
%%
% After downloading the data from GitHub, unzip the file in your temporary directory.
tempdir = 'C:\Users\eshchekinova\Documents\BoknisData\GoogleClass';
%load(fullfile(tempdir,'FiguresDataBoknis','DataBoknis.mat'))
%load(fullfile(tempdir,'SpectroBoknisData_v1','BoknisDataSpectro.mat'))
load(fullfile(tempdir,'SpectroBoknisData','BoknisDataSpectro.mat'))
%%
% |ECGData| is a structure array with two fields: |Data| and |Labels|. The
% |Data| field is a 162-by-65536 matrix where each row is an ECG recording
% sampled at 128 hertz. |Labels| is a 162-by-1 cell array of diagnostic
% labels, one for each row of |Data|. The three diagnostic categories are:
% 'ARR', 'CHF', and 'NSR'.
%%
% To store the preprocessed data of each category, first create an ECG data
% directory |dataDir| inside |tempdir|. Then create three subdirectories in
% |'data'| named after each ECG category. The helper function
% |helperCreateECGDirectories| does this. |helperCreateECGDirectories|
% accepts |ECGData|, the name of an ECG data directory, and the name of a
% parent directory as input arguments. You can replace |tempdir| with
% another directory where you have write permission. You can find the
% source code for this helper function in the Supporting Functions section
% at the end of this example.
parentDir = tempdir;
dataDir = 'SpectroBoknisData';
helperCreateDirectories(BoknisDataSpectro,parentDir,dataDir)
% dataDir = 'FiguresDataBoknis';
% helperCreateDirectories(DataBoknis,parentDir,dataDir)
%%
% Plot a representative of each ECG category. The helper function
% |helperPlotReps| does this. |helperPlotReps| accepts |ECGData| as input. 
% You can find the source code for this helper function in the Supporting
% Functions section at the end of this example.
helperPlotReps(BoknisDataSpectro)
%helperPlotReps(DataBoknis)
%% Create Time-Frequency Representations
% After making the folders, create time-frequency representations of the
% ECG signals. These representations are called scalograms. A scalogram is
% the absolute value of the CWT coefficients of a signal.
%
% To create the scalograms, precompute a CWT filter bank. Precomputing the
% CWT filter bank is the preferred method when obtaining the CWT of many
% signals using the same parameters.
%
% Before generating the scalograms, examine one of them. Create a CWT
% filter bank using
% <docid:wavelet_ref#mw_b8b0ddad-1703-45db-a219-febab76b6d1f cwtfilterbank>
% for a signal with 1000 samples. Use the filter bank to take the CWT of
% the first 1000 samples of the signal and obtain the scalogram from the
% coefficients.
% delT=new_time_sec(3)-new_time_sec(2);
% Fs = 1/delT;
% time_length=length(DataBoknis.Data(1,:,1))
% fb = cwtfilterbank('SignalLength',time_length,...
%     'SamplingFrequency',Fs,...
%     'VoicesPerOctave',16);
% sig = DataBoknis.Data(1,1:time_length,1);
% [cfs,frq] = wt(fb,sig);
% t = (0:time_length-1)/Fs;
% figure;
% size(abs(cfs))
% length(t)
% pcolor(t,frq,abs(cfs))
% set(gca,'yscale','log');shading interp;axis tight;
% title('Scalogram');xlabel('Time (s)');ylabel('Frequency (Hz)')
%%
% Use the helper function, |helperCreateRGBfromTF|, to create the
% scalograms as RGB images and write them to the appropriate subdirectory
% in |dataDir|. The source code for this helper function is in the
% Supporting Functions section at the end of this example. To be compatible
% with the GoogLeNet architecture, each RGB image is an array of size
% 224-by-224-by-3.
%helperCreateRGBfromTF_Wavelet(new_time_sec,BoknisDataSpectro,parentDir,dataDir)
%helperCreateRGBfromTF_v0(new_time_sec,BoknisDataSpectro2019,parentDir,dataDir)
%% Divide into Training and Validation Data
% Load the scalogram images as an image datastore. The |imageDatastore|
% function automatically labels the images based on folder names and stores
% the data as an ImageDatastore object. An image datastore enables you to
% store large image data, including data that does not fit in memory, and
% efficiently read batches of images during training of a CNN.
allImages = imageDatastore(fullfile(parentDir,dataDir),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
%%
% Randomly divide the images into two groups, one for training and the
% other for validation. Use 80% of the images for training, and the
% remainder for validation. For purposes of reproducibility, we set the
% random seed to the default value.
rng default
[imgsTrain,imgsValidation] = splitEachLabel(allImages,0.8,'randomized');
disp(['Number of training images: ',num2str(numel(imgsTrain.Files))]);
disp(['Number of validation images: ',num2str(numel(imgsValidation.Files))])
%% GoogLeNet
%
% *Load*
%
% Load the pretrained GoogLeNet neural network. If Deep Learning
% Toolbox(TM) Model _for GoogLeNet Network_ support package is not
% installed, the software provides a link to the required support package
% in the Add-On Explorer. To install the support package, click the link,
% and then click *Install*.
net = googlenet;
%%
% Extract the layer graph from the network and plot the layer graph.
lgraph = layerGraph(net);
numberOfLayers = numel(lgraph.Layers);
figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
plot(lgraph)
title(['GoogLeNet Layer Graph: ',num2str(numberOfLayers),' Layers']);
%%
% Inspect the first element of the network Layers property. Notice that
% GoogLeNet requires RGB images of size 224-by-224-by-3.
net.Layers(1)
%%
% *Modify GoogLeNet Network Parameters*
%
% Each layer in the network architecture can be considered a filter. The
% earlier layers identify more common features of images, such as blobs,
% edges, and colors. The later layers focus on more specific features in
% order to differentiate categories.
%
% To retrain GoogLeNet to our ECG classification problem, replace the last
% four layers of the network. The first of the four layers,
% |'pool5-drop_7x7_s1'| is a dropout layer. A dropout layer randomly sets
% input elements to zero with a given probability. The dropout layer is
% used to help prevent overfitting. The default probability is 0.5. See
% <docid:nnet_ref#mw_9d3adffa-df2e-40f8-9f15-972965a8d78d dropoutLayer> for
% more information. The three remaining layers, 'loss3-classifier', 'prob',
% and 'output', contain information on how to combine the features that the
% network extracts into class probabilities and labels. By default, the
% last three layers are configured for 1000 categories.
%
% Add four new layers to the layer graph: a dropout layer with a
% probability of 60% dropout, a fully connected layer, a softmax layer, and
% a classification output layer. Set the final fully connected layer to
% have the same size as the number of classes in the new data set (3, in
% this example). To learn faster in the new layers than in the transferred
% layers, increase the learning rate factors of the fully connected layer.
% Store the GoogLeNet image dimensions in |inputSize|.
lgraph = removeLayers(lgraph,{'pool5-drop_7x7_s1','loss3-classifier','prob','output'});
categories(imgsTrain.Labels)
numClasses = numel(categories(imgsTrain.Labels))
newLayers = [
    dropoutLayer(0.6,'Name','newDropout')
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',5,'BiasLearnRateFactor',5)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);

lgraph = connectLayers(lgraph,'pool5-7x7_s1','newDropout');
inputSize = net.Layers(1).InputSize;
%%
% *Set Training Options and Train GoogleNet*
%
% Training a neural network is an iterative process that involves
% minimizing a loss function. To minimize the loss function, a gradient
% descent algorithm is used. In each iteration, the gradient of the loss
% function is evaluated and the descent algorithm weights are updated.
% 
% Training can be tuned by setting various options. |InitialLearnRate|
% specifies the initial step size in the direction of the negative gradient
% of the loss function. |MiniBatchSize| specifies how large of a subset of the
% training set to use in each iteration. One epoch is a full pass of the
% training algorithm over the entire training set. |MaxEpochs| specifies
% the maximum number of epochs to use for training. Choosing the right
% number of epochs is not a trivial task. Decreasing the number of epochs
% has the effect of underfitting the model, and increasing the number of
% epochs results in overfitting.
%
% Use the <docid:nnet_ref#bu59f0q trainingOptions> function to
% specify the training options. Set |MiniBatchSize| to 10, |MaxEpochs| to
% 10, and |InitialLearnRate| to 0.0001. Visualize training progress by
% setting |Plots| to |training-progress|. Use the stochastic gradient
% descent with momentum optimizer. By default, training is done on a GPU if
% one is available (requires Parallel Computing Toolbox(TM) and a CUDA&reg;
% enabled GPU with compute capability 3.0 or higher). For purposes of
% reproducibility, train the network using only one CPU, by setting the
% |ExecutionEnvironment| to |cpu|, and set the random seed to the default
% value. Run times will be faster if you are able to use a GPU.

options = trainingOptions('sgdm',...
    'MiniBatchSize',15,...
    'MaxEpochs',20,...
    'InitialLearnRate',1e-4,...
    'ValidationData',imgsValidation,...
    'ValidationFrequency',10,...
    'ValidationPatience',Inf,...
    'Verbose',1,...
    'ExecutionEnvironment','cpu',...
    'Plots','training-progress');
%%
% The training process usually takes 1-5 minutes on a desktop CPU. The
% command window displays training information during the run. Results will
% include epoch number, iteration number, time elapsed, mini-batch
% accuracy, validation accuracy, and loss function value for the validation
% data.
rng default
trainedGN = trainNetwork(imgsTrain,lgraph,options);
%%
% Inspect the last three layers of the trained network. Note that the
% Classification Output Layer mentions the three labels.
trainedGN.Layers(end-2:end)
cNames = trainedGN.Layers(end).ClassNames
%%
% *Evaluate GoogLeNet Accuracy*
%
% Evaluate the network using the validation data.
[YPred,probs] = classify(trainedGN,imgsValidation);
accuracy = mean(YPred==imgsValidation.Labels);
display(['GoogLeNet Accuracy: ',num2str(accuracy)])
%%
% The accuracy is identical to the validation accuracy reported on the
% training visualization figure. The scalograms were split into training
% and validation collections. Both collections were used to train
% GoogLeNet. The ideal way to evaluate the result of the training is to
% have the network classify data it has not seen. Since there is an
% insufficient amount of data to divide into training, validation, and
% testing, we treat the computed validation accuracy as the network
% accuracy.
% %%
% % *Explore GoogLeNet Activations*
% %
% % Each layer of a CNN produces a response, or activation, to an input
% % image. However, there are only a few layers within a CNN that are
% % suitable for image feature extraction. The layers at the beginning of the
% % network capture basic image features, such as edges and blobs. To see
% % this, visualize the network filter weights from the first convolutional
% % layer. There are 64 individual sets of weights in the first layer.
% wghts = trainedGN.Layers(2).Weights;
% wghts = rescale(wghts);
% wghts = imresize(wghts,5);
% figure
% %montage(wghts)
% createImMontage(wghts)
% title('First Convolutional Layer Weights')
% %%
% % You can examine the activations and discover which features GoogLeNet
% % learns by comparing areas of activation with the original image. For more
% % information, see <docid:nnet_examples#bvl0ibp Visualize Activations of a
% % Convolutional Neural Network> and <docid:nnet_examples#bvm8dx7 Visualize
% % Features of a Convolutional Neural Network>.
% %
% % Examine which areas in the convolutional layers activate on an image from
% % the |ARR| class. Compare with the corresponding areas in the
% % original image. Each layer of a convolutional neural network consists of
% % many 2-D arrays called _channels_. Pass the image through the network
% % and examine the output activations of the first convolutional layer,
% % |'conv1-7x7_s2'|.
% convLayer = 'conv1-7x7_s2';
% 
% % imgClass = 'YEVENT';
% % imgName = 'YEVENT_36.jpg';
% imgClass = 'EVENT';
% imgName = 'EVENT_60.jpg';
% imarr = imread(fullfile(parentDir,dataDir,imgClass,imgName));
% 
% trainingFeatures = activations(trainedGN,imarr,convLayer);
% sz = size(trainingFeatures);
% trainingFeatures = reshape(trainingFeatures,[sz(1) sz(2) 1 sz(3)]);
% % figure
% % montage(rescale(trainingFeatures),'Size',[8 8])
% % title([imgClass,' Activations'])
% %%
% % Find the strongest channel for this image. Compare the strongest channel
% % with the original image.
% imgSize = size(imarr);
% imgSize = imgSize(1:2);
% [~,maxValueIndex] = max(max(max(trainingFeatures)));
% arrMax = trainingFeatures(:,:,:,maxValueIndex);
% arrMax = rescale(arrMax);
% arrMax = imresize(arrMax,imgSize);
% % figure;
% % imshowpair(imarr,arrMax,'montage')
% % title(['Strongest ',imgClass,' Channel: ',num2str(maxValueIndex)])
%% AlexNet
%
% AlexNet is a deep CNN whose architecture supports images of size
% 227-by-227-by-3. Even though the image dimensions are different for
% GoogLeNet, you do not have to generate new RGB images at the AlexNet
% dimensions. You can use the original RGB images.
%
% *Load*
%
% Load the pretrained AlexNet neural network. If Deep Learning Toolbox(TM)
% Model _for AlexNet Network_ support package is not installed, the
% software provides a link to the required support package in the Add-On
% Explorer. To install the support package, click the link, and then click
% *Install*.
alex = alexnet;
%%
% Review the network architecture. Note that the first layer specifies the
% image input size as 227-by-227-by-3, and that AlexNet has fewer layers than
% GoogLeNet.
layers = alex.Layers
%%
% *Modify AlexNet Network Parameters*
%
% To retrain AlexNet to classify new images, make changes similar to those
% made for GoogLeNet.
%
% By default, the last three layers of AlexNet are configured for 1000
% categories. These layers must be fine-tuned to our ECG classification
% problem. Layer 23, the fully connected layer, must be set to have the
% same size as the number of categories in our ECG data. Layer 24 does not
% need to change with our ECG classification problem. |Softmax| applies a
% softmax function to the input. See
% <docid:nnet_ref#mw_a09d3c68-d062-4692-a950-9a7fea5c40c3 SoftmaxLayer> for
% more information. Layer 25, the Classification Output layer, holds the
% name of the loss function used for training the network and the class
% labels. Since there are three ECG categories, set layer 23 to be a fully
% connected layer of size equal to 3, and set layer 25 to be the
% classification output layer.
layers(23) = fullyConnectedLayer(2);
layers(25) = classificationLayer;
%%
% *Prepare RGB Data for AlexNet*
%
% The RGB images have dimensions appropriate for the GoogLeNet
% architecture. Obtain from the first AlexNet layer the image dimensions
% used by AlexNet. Use those dimensions to create augmented image
% datastores that will automatically resize the existing RGB images for the
% AlexNet architecture. For more information, see
% <docid:nnet_ref#mw_fd462475-1bc2-4f76-968c-c75721e5195f
% augmentedImageDatastore>.
inputSize = alex.Layers(1).InputSize;
augimgsTrain = augmentedImageDatastore(inputSize(1:2),imgsTrain);
augimgsValidation = augmentedImageDatastore(inputSize(1:2),imgsValidation);
%%
% *Set Training Options and Train AlexNet*
%
% Set the training options to match those used for GoogLeNet. Then train
% AlexNet. The training process usually takes 1-5 minutes on a desktop CPU.
rng default
mbSize = 10;
mxEpochs = 10;
ilr = 1e-4;
plt = 'training-progress';

opts = trainingOptions('sgdm',...
    'InitialLearnRate',ilr, ...
    'MaxEpochs',mxEpochs ,...
    'MiniBatchSize',mbSize, ...
    'ValidationData',augimgsValidation,...
    'ExecutionEnvironment','cpu',...
    'Plots',plt);

trainedAN = trainNetwork(augimgsTrain,layers,opts);
%%
% The validation accuracy is 93.75%. Inspect the last three layers of the
% trained AlexNet network. Observe the Classification Output Layer mentions
% the three labels.
trainedAN.Layers(end-2:end)


% % Conclusion
% This example shows how to use transfer learning and continuous wavelet
% analysis to classify three classes of ECG signals by leveraging the
% pretrained CNNs GoogLeNet and AlexNet. Wavelet-based time-frequency
% representations of ECG signals are used to create scalograms. RGB images
% of the scalograms are generated. The images are used to fine-tune both
% deep CNNs. Activations of different network layers were also explored.
% 
% This example illustrates one possible workflow you can use for
% classifying signals using pretrained CNN models. Other workflows are
% possible. GoogLeNet and AlexNet are models pretrained on a subset of the
% ImageNet database [9], which is used in the ImageNet Large-Scale Visual
% Recognition Challenge (ILSVRC) [10]. The ImageNet collection contains
% images of real-world objects such as fish, birds, appliances, and fungi.
% Scalograms fall outside the class of real-world objects. In order to fit
% into the GoogLeNet and AlexNet architecture, the scalograms also
% underwent data reduction. Instead of fine-tuning pretrained CNNs to
% distinguish different classes of scalograms, training a CNN from scratch
% at the original scalogram dimensions is an option.
% % References
% # Baim DS, Colucci WS, Monrad ES, Smith HS, Wright RF, Lanoue A, Gauthier DF, Ransil BJ, Grossman W, Braunwald E. Survival of patients with severe congestive heart failure treated with oral milrinone. J American College of Cardiology 1986 Mar; 7(3):661-670.
% # Engin, M., 2004. ECG beat classification using neuro-fuzzy network. Pattern Recognition Letters, 25(15), pp.1715-1722.
% # Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit,and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220 [Circulation Electronic Pages; http://circ.ahajournals.org/content/101/23/e215.full]; 2000 (June 13).
% # Leonarduzzi, R.F., Schlotthauer, G., and Torres. M.E. 2010. Wavelet leader based multifractal analysis of heart rate variability during  myocardial ischaemia. Engineering in Medicine and Biology Society (EMBC), 2010 Annual International Conference of the IEEE.
% # Li, T. and Zhou, M., 2016. ECG classification using wavelet packet entropy and random forests. Entropy, 18(8), p.285.
% # Maharaj, E.A. and Alonso, A.M. 2014. Discriminant analysis of multivariate time series: Application to diagnosis based on ECG signals. Computational Statistics and Data Analysis, 70, pp. 67-87.
% # Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)
% # Zhao, Q. and Zhang, L., 2005. ECG feature extraction and classification using wavelet transform and support vector machines. IEEE International Conference on Neural Networks and Brain,2, pp. 1089-1092.
% # _ImageNet_. http://www.image-net.org
% # Russakovsky, O., Deng, J., Su, H., et al. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV). Vol 115, Issue 3, 2015, pp. 211-252.
% % Supporting Functions
% *helperCreateECGDataDirectories* Creates a data directory inside a parent
% directory, then creates three subdirectories inside the data directory.
% The subdirectories are named after each class of ECG signal found in
% |ECGData|.
% 
% <include>helperCreateECGDirectories.m</include>
% 
% %
% *helperPlotReps* Plots the first thousand samples of a representative of
% each class of ECG signal found in |ECGData|.
% 
% <include>helperPlotReps.m</include>
% 
% %
% *helperCreateRGBfromTF* Uses *cwtfilterbank* to obtain the continuous
% wavelet transform of the ECG signals and generates the scalograms from
% the wavelet coefficients. The helper function resizes the scalograms and
% writes them to disk as jpeg images.
% 
% <include>helperCreateRGBfromTF.m</include>
% 
% 
% 
dirNetwork = 'C:\Users\eshchekinova\Documents\BoknisData\GoogleClass\Networks';
cd(dirNetwork);
save AlexGoogleNetsTrained_Wavelet trainedAN trainedGN
cd(tempdir);
