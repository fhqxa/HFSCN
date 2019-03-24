function [Model_HierSVM,TimeTestMean,RealLabel,OutLabel,AccMean,AccStd,F_LCAMean,FHMean,TIEMean] = FeaSelc_K_Fold_SVM_hierarchical_classifier_lxx(data_array, tree, FeaIdx, nFeaSel, Kfold, svm_opt)
    indexRoot = tree_Root(tree);
    internalNodes    = tree_InternalNodes(tree);
    n_internalNodes  = length(internalNodes);
    
    [M,~]=size(data_array);
    Acc = zeros(1,Kfold);
    rand('seed',1);
    indices = crossvalind('Kfold',M,Kfold);
    for k = 1:Kfold
        testID = (indices == k);    % test集元素在数据集中对应的单元编号
        trainID = ~testID;          % train集元素的编号为非test元素的编号        
        train_array = data_array(trainID,:);
        test_array = data_array(testID,:);
        
        [X, Y] = creatSubTable(train_array, tree);
        
        Fea_Sel_id = FeaIdx{indexRoot}(1:nFeaSel);        
        x = X{indexRoot}(:,Fea_Sel_id);
        y = Y{indexRoot};
        Model_HierSVM{indexRoot} = svmtrain(y,x,svm_opt);
        for i = 1:n_internalNodes   
            i_node = internalNodes(i);
            if (isempty(Y{i_node})==0)
                Fea_Sel_id = FeaIdx{i_node}(1:nFeaSel);
                x = X{i_node}(:,Fea_Sel_id);
                y = Y{i_node};
                Model_HierSVM{i_node} = svmtrain(y,x,svm_opt);
            end
        end
        
        test_data = test_array(:,1:end-1);
        test_label = test_array(:,end); RealLabel{1,k} = test_label;
        [n_test,~] = size(test_array);
        outlabel = zeros(n_test,1);
        
        tic;
        for i=1:n_test
            Fea_Sel_id = FeaIdx{indexRoot}(1:nFeaSel);
            x = test_data(i,Fea_Sel_id);
            y = test_label(i,1);
            [outlabel(i,1),~, ~] = svmpredict(y,x,Model_HierSVM{indexRoot},'-q');
            while ismember(outlabel(i,1),internalNodes)
                Fea_Sel_id = FeaIdx{outlabel(i,1)}(1:nFeaSel);
                x = test_data(i,Fea_Sel_id);
                y = outlabel(i,1);
                [outlabel(i,1),~, ~] = svmpredict(y,x,Model_HierSVM{outlabel(i,1)},'-q');
            end
        end
        TimeTest(1,k) = toc;
        
        OutLabel{1,k} = outlabel;
        [PH(1,k), RH(1,k), FH(1,k)] = EvaHier_HierarchicalPrecisionAndRecall(test_label,outlabel',tree);
        [P_LCA(1,k),R_LCA(1,k),F_LCA(1,k)] = EvaHier_HierarchicalLCAPrecisionAndRecall(test_label,outlabel',tree);
        TIE(1,k) = EvaHier_TreeInducedError(test_label,outlabel',tree);     
        Acc(1,k) = length(find((test_label - outlabel)==0))/length(outlabel);
    end
    TimeTestMean = mean(TimeTest);
    AccMean = mean(Acc);  
    AccStd = std(Acc);
    F_LCAMean = mean(F_LCA);
    FHMean = mean(FH);
    TIEMean = mean(TIE);  
end