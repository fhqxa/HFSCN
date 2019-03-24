%% HFSCN£ºcapped least squared regression + parent-child relationship
% Obj = ||XW-Y||^T*F*||XW-Y||+||W||_2,1+||W-W_pi||_2^2
% Input:
%     X - the data matrix without the label
%     Y - labels
%     lambda - the parameter of optimal ||W||_2,1
%     alpha - the tradeoff parameter of parent-child relationship
%     epsi  - the portation of outliers which should be setted by 
%     flag - draw the objective value
% Output:
%     Outliers     - the first column is the threshold value of XW_Y; 
%                    the second column is the number of outliers.
%     feature_slct - The selected feature subset.

%% Function
function [Outliers,feature_slct] = HFSCN(X, Y, tree, lambda, alpha, epsi, flag) 
    internalNodes = tree_InternalNodes(tree);
    indexRoot = tree_Root(tree);% The root of the tree
    noLeafNode =[internalNodes;indexRoot];    
    numSelected = size(X{indexRoot},2);
    Outliers = zeros(length(tree),2);% the number of outliers, and the threshold value
    maxIte = 10;
    for i = 1:length(noLeafNode)
        ClassLabel = unique(Y{noLeafNode(i)});
        m(noLeafNode(i)) = length(ClassLabel);
    end
    maxm = max(m);
    %% initialize
    for j = 1:length(noLeafNode)
        [~,n] = size(X{noLeafNode(j)}); % get the number of features
        Y{noLeafNode(j)} = conversionY01_extend(Y{noLeafNode(j)},maxm);% extend 2 to [1 0]
        W{noLeafNode(j)} = ones(n, maxm);
    end
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:maxIte
        %% initialization on each non-LeafNode
        for j = 1:length(noLeafNode)
            D{noLeafNode(j)} = diag(0.5./max(sqrt(sum(W{noLeafNode(j)}.*W{noLeafNode(j)},2)),eps));
            mSample = size(X{noLeafNode(j)},1);
            if epsi< 10^-8
                F{noLeafNode(j)} = eye(mSample);
            else
                WX_Y_N = sum((X{noLeafNode(j)}* W{noLeafNode(j)} -Y{noLeafNode(j)}).^2,2).^(1/2);                
                temp = sort(WX_Y_N);
                epsi2 = temp(round((1-epsi) * mSample));
                Outliers(noLeafNode(j),1) = mSample - round((1-epsi) * mSample);
                Outliers(noLeafNode(j),2) = epsi2;
                Ind = (WX_Y_N <= epsi2);                
                F{noLeafNode(j)} = diag(1./(WX_Y_N) .* Ind);
            end
            XFX{noLeafNode(j)} = X{noLeafNode(j)}'* F{noLeafNode(j)} * X{noLeafNode(j)};
            XFY{noLeafNode(j)} = X{noLeafNode(j)}'* F{noLeafNode(j)} * Y{noLeafNode(j)};
        end
        %% Update the root node
        W_interal = zeros(n,maxm);
        childofRoot = find(tree(:,1)==find(tree(:,1)==0));
        leafNode = tree_LeafNode(tree);
        childofRoot = setdiff(childofRoot,leafNode); % delete the leaf node.
        
        for j = 1:length(childofRoot)
            W_interal =  W_interal + W{childofRoot(j)};
        end
        W{indexRoot} = (XFX{indexRoot} + lambda * D{indexRoot} + alpha *length(childofRoot) * eye(n)) \ (XFY{indexRoot} + alpha * W_interal );
        
        %% Update the internal nodes
        for j = 1:length(internalNodes)
            currentNodeParent = tree(internalNodes(j),1);
            W{internalNodes(j)} = (XFX{internalNodes(j)} + lambda * D{internalNodes(j)} + alpha * eye(n)) \ (XFY{internalNodes(j)} + alpha * W{currentNodeParent});        
        end   
        %% The value of object function
        if (flag ==1)
            obj(i)=trace((X{indexRoot}*W{indexRoot}-Y{indexRoot})'*F{indexRoot}*(X{indexRoot}*W{indexRoot}-Y{indexRoot}))+lambda*L21(W{indexRoot});
            for j = 1:length(internalNodes)
                currentNodeParent = tree(internalNodes(j),1);
                obj(i)=obj(i)+ trace((X{internalNodes(j)}*W{internalNodes(j)}-Y{internalNodes(j)})'*F{internalNodes(j)}*(X{internalNodes(j)}*W{internalNodes(j)}-Y{internalNodes(j)}));
                obj(i)=obj(i)+lambda*L21(W{internalNodes(j)})+ alpha * norm(W{internalNodes(j)} - W{currentNodeParent})^2;
            end
        end
    end
    %% Objective vlue
    for i = 1:length(noLeafNode)
        W1=W{noLeafNode(i)};
        W{noLeafNode(i)} = W1(:,1:m(noLeafNode(i)));
    end

    clear W1;
    for j = 1: length(noLeafNode)
        tempVector = sum(W{noLeafNode(j)}.^2, 2);
        [atemp, value] = sort(tempVector, 'descend'); % sort tempVecror (W) in a descend order
        clear tempVector;
        feature_slct{noLeafNode(j)} = value(1:numSelected);
    end
    if (flag == 1)
        figure;
        set(gcf,'color','w');
        plot(obj,'LineWidth',4,'Color',[0 0 1]);
        set(gca,'FontName','Times New Roman','FontSize',18);
        xlabel('Iteration number');
        ylabel('Objective function value');
    end
end