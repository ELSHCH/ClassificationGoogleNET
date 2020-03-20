function helperPlotReps(GData)
% This function is only intended to support the ECGAndDeepLearningExample.
% It may change or be removed in a future release.

folderLabels = unique(GData.Labels)
Dl=length(GData.Data(1,:));
for k=1:2
    eventType = folderLabels{k};
    ind = find(ismember(GData.Labels,eventType));
    subplot(2,1,k)
    plot(GData.Data(ind(1),1:Dl));
    grid on
    title(eventType)
end
end
