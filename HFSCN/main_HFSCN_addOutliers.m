%% the main function of HFSCN 
% writed by Xinxin Liu 
% Updated at 2019-03-11
 
% %% add outliers to the training set; creatsubtable; save the data file (X, Y, tree)
% clc;clear;close all
% load Car196;   % 'DD27';'Protein194';'VOC20';'Car196'
% ClassLabel = data_array(:,end);
% [M,N] = size(data_array);
% P_outlier = 0.10;    % outlier percentage
% N_outlier = round( M * P_outlier);
% rand('seed',1); 
% Fea_outlier = 10 * rand(N_outlier, N-1);
% Label_outlier = randi([min(ClassLabel), max(ClassLabel)],N_outlier,1);
% data_array = [data_array;Fea_outlier,Label_outlier];
% [X, Y] = creatSubTable(data_array, tree);
% save Car196TrainSubTable_outlier X Y tree;


%% Main Function
clear;clc;close all;
str1={'Car196'};% 'DD27';'Protein194';'VOC20';'Car196';'ILSVRC57'; 'CLEF';'Cifar4096d';'Sun324'};
m = length(str1);
for iDataset = 1:m
    DataName = str1{iDataset};
    load ([DataName 'TrainSubTable_outlier']);
    numFolds = 10;    
    flag = 1;%  %  wether figure the objective value
    svm_opt = '-s 0 -c 1 -t 0 -q';
    
    k=1;
    for iLam =2%-2:1:2
        lamda = 10^iLam;%10^iLam;
        for iAlp = 1:2%-2:1:2
            alpha = 10^iAlp;%10^iAlp;
            Epsi_all = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.25];
            for iEpsi = 9%1:15
                epsi = Epsi_all(iEpsi);
               %% feature selection
                tic;
                [nOutliers, feature] = HFSCN(X, Y, tree, lamda, alpha, epsi, flag);
                t_TrainFeature = toc;
                save([str1{iDataset} 'Trainfeature_RHFS_' num2str(iLam) '_' num2str(iAlp) '_' num2str(iEpsi) '_outlier'],'feature','nOutliers','t_TrainFeature');
               
               %% Test feature
                load([DataName 'Test']);%Test']);
                data_array = double(data_array); N_feature = size(data_array,2)-1; 
                for nPart = 20%5:5:30
                    nFeaSel = round(nPart* 0.01 * N_feature);
                    tic;
                    [Model_HierSVM,TimeTestMean,RealLabel,OutLabel,AccMean,AccStd,F_LCAMean,FHMean,TIEMean] = FeaSelc_K_Fold_SVM_hierarchical_classifier_lxx(data_array, tree, feature, nFeaSel, numFolds, svm_opt);
                    t_TestFeature = toc;
                    Results(k,1) = lamda;
                    Results(k,2) = alpha;
                    Results(k,3) = epsi;
                    Results(k,4) = nFeaSel;
                    Results(k,5:11) = [AccMean,AccStd,F_LCAMean,FHMean,TIEMean,t_TestFeature,t_TrainFeature];
                    k=k+1;
                end
            end
        end
    end
    save([DataName '_RHFS_SVM_results_outlier'], 'Results');
    %% 保存结果到excel文件
    [mR, nR] = size(Results);          
    Results_cell = mat2cell(Results, ones(mR,1), ones(nR,1));% 将Results切割成m*n的cell矩阵
    title = {'Lambda' 'Alpha' 'Epsi' 'nFeatures' 'Acc' 'AccStd' 'F_LCA' 'FH' 'TIE' 't_test' 't_train'}; % 添加变量名称
    s = xlswrite([DataName '_RHFS_SVM_results_outlier.xls'], [title; Results_cell]); 
end
fprintf('Lambda Alpha Epsi nFeatures AccMean AccStd F_LCAMean FHMean TIEMean t_TestFeature t_TrainFeature\n');
DataName
fprintf('---- HFSCN_addOutliers completed ----\n');
