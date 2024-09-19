classdef FeatureInstanceSelection < PROBLEM
% <multi> <real/binary> <none> 
% The feature and instance selection problem with constraints
% dataNo --- 20 --- Index of dataset
% encode --- 1 --- Encoding method
% ObjectiveNo --- 3 --- Number of objectives
% NoiseLevel --- 0 --- Percentage(%) noise added 

%------------------------------- Reference --------------------------------
% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% The datasets are taken from the UCI machine learning repository: http://archive.ics.uci.edu/ml/index.php
%  No.    Name                              Samples Features Classes   MinorityClass(%) MajorityClass(%) ImbalanceRatio
%  1      Glass                               214      10       6          4.21              35.51          8.44
%  2      Wine                                178      13       3          26.97             39.89          1.48
%  3      Leaf                                340      14      30          2.35              4.71           2
%  4      Australian                          690      14       2          44.49             55.51          1.25
%  5      Zoo                                 101      17       7          3.96              40.59          10.25
%  6      Lymph                               148      18       4          1.35              54.73          40.50
%  7      Vehicle                             846      18       4          23.52             25.77          1.10
%  8      ImageSegmentation                   210      19       7          14.29             14.29          1
%  9      Parkinson                           195      22       2          24.62             75.38          3.06
% 10      Spect                               267      22       2          41.20             58.80          1.43
% 11      German                             1000      24       2          30                70             2.33
% 12      LedDisplay                         1000      24      10          8.3               11.6           1.4
% 13      WallRobot                          5456      24       4          6.01              40.41          6.72
% 14      WBCD                                569      30       2          37.26             62.74          1.68
% 15      GuesterPhase                       9873      32       5          10.11             29.88          2.96
% 16      Dermatology                         366      33       6          5.46              30.60          5.60
% 17      Ionosphere                          351      34       2          35.9              64.1           1.79
% 18      Chess                              3196      36       2          47.78             52.22          1.09
% 19      Connect4                          67557      42       3          9.55              65.83          6.90
% 20      Lung                                 32      56       2          28.13             71.88          2.56
% 21      Sonar                               208      60       2          46.63             53.37          1.14
% 22      Plant                              1600      64     100          1                 1              1
% 23      Mice                               1077      77       8          9.75              13.93          1.43
% 24      Movementlibras                      360      90      15          6.67              6.67           1
% 25      Hillvalley                         1212     100       2          49.51             50.5           1.02
% 26      GasSensor                         13910     128       6          11.8              21.63          1.83
% 27      Musk1                               476     166       2          43.49             56.51          1.3
% 28      USPS                               9298     256      10          7.61              16.7           2.19
% 29      Semeion                            1593     265       2          9.92              90.08          9.08
% 30      Arrhythmia                          452     278      13          0.44              54.2           122.5
% 31      LSVT                                126     310       2          33.33             66.67          2
% 32      lung_discrete                        73     325       7          6.85              28.77          4.2
% 33      Madelon                            2600     500       2          50                50             1
% 34      Isolet                             1560     617      26          3.85              3.85           1
% 35      Isolet5                            1559     617      25          3.78              3.85           1.02
% 36      MultipleFeatures                   2000     649      10          10                10             1
% 37      Gametes                            1600    1000       2          50                50             1
% 38      ORL                                 400    1024      40          2.5               2.5            1 
% 39      Yale                                165    1024      16          6.67              6.67           1
% 40      COIL20                             1440    1024      20          5                 5              1
% 41      Christine                          5418    1636       2          50                50             1
% 42      Bioresponse                        3751    1776       2          45.77             54.23          1.18
% 43      colon                                62    2000       2          35.38             64.52          1.82
% 44      Colon                                62    2000       2          35.48             64.52          1.82
% 45      SRBCT                                83    2308       4          13.25             34.94          2.64
% 46      AR10P                               130    2400      10          10                10             1 
% 47      lymphoma                             96    4026       9          2.08              47.92          23
% 48      RELATHE                            1427    4322       2          45.41             54.59          1.2
% 49      GLIOMA                               50    4434       4          14                30             2.14
% 50      BASEHOCK                           1993    4862       2          49.87             50.13          1.01
% 51      Gisette                            7000    5000       2          50                50             1
% 52      Leukemia1                            72    5327       3          12.5              52.78          4.22
% 53      DLBCL                                77    5469       2          24.68             75.32          3.05
% 54      9Tumor                               60    5726       9          3.33              15             4.5
% 55      Brain1                               90    5920       5          4.44              66.67          15
% 56      Prostate-GE                         102    5966       2          49.02             50.98          1.04
% 57      CNS                                  60    7129       2          35                65             1.86
% 58      Leukemia                             72    7129       2          34.72             65.28          1.88
% 59      ALLAML                               72    7129       2          34.72             65.28          1.88 
% 60      Carcinom                            174    9182      11          3.45              15.52          4.5
% 61      nci9                                 60    9712       9          3.33              15             1.5
% 62      arcene                              200   10000       2          44                56             1.27
% 63      orlraws10P                          100   10304      10          10                10             1
% 64      Brain2                               50   10367       4          14                30             2.14
% 65      CLL-SUB-111                         111   11340       3          9.91              45.95          4.64
% 66      11Tumor                             174   12533      11          3.45              15.52          4.5
% 67      SMK-CAN-187                         187   19993       2          48.13             51.87          1.08
% 68      GLI-85                               85   22283       2          30.59             69.41          2.27
% 69      Bhattacharjee-2001                  203   1543        5          2.96              68.47          23.17
% 70      Gordon-2002                         181   1626        2          17.13             82.87          4.84
% 71      Yeoh-2002-v1                        248   2526        2          17.34             82.66          4.77

    properties(Access = private)
        TestIn;             % Input of test set
        TestOut;            % Output of test set
        Category;           % Output label set
        dataNo;auto
        TestNum;
        K;                  % The number of nearest neighbor in KNN
        FullErrorRate;     % Claasification error ratio using all features and instances
        FullMinErrorRate;  % Worst-class claasification error ratio  using all features and instances
        PopWorstError;
        PopError;
    end

    properties(Access = public)
        boolTrain;  % 1 means apply the data to the training set, otherwise to the test set
        TrainIn;    % Input of training set
        TrainOut;   % Output of training set
    end

    methods
        %% Default settings of the problem
        function Setting(obj)
            [dataNo, encode, objectiveNo, NoiseLevel] = obj.ParameterSet(1, 1, 3, 0);
            addpath(genpath('FSdata'));
            %% Load data
            str = {'Glass.mat',      'Wine.mat',          'Leaf.mat',              'Australian.mat',       'Zoo.mat',...
                'Lymph.mat',            'Vehicle.mat',       'ImageSegmentation.mat', 'Parkinson.mat',        'Spect.mat',...
                'German.mat',           'LedDisplay.mat',    'WallRobot.mat',         'WBCD.mat',             'GuesterPhase.mat',...
                'Dermatology.mat',      'Ionosphere.mat',    'Chess.mat',             'Connect4.mat',         'Lung.mat',...
                'Sonar.mat',            'Plant.mat',         'Mice.mat',              'Movementlibras.mat',   'Hillvalley.mat',...
                'GasSensor.mat',        'Musk1.mat',         'USPS.mat',              'Semeion.mat',          'Arrhythmia.mat',...
                'LSVT.mat',             'lung_discrete.mat', 'Madelon.mat',           'Isolet.mat',           'Isolet5.mat',...
                'MultipleFeatures.mat', 'Gametes.mat',       'ORL.mat',               'Yale.mat',             'COIL20.mat',...
                'Christine.mat',        'Bioresponse.mat',   'colon.mat',             'Colon.mat',            'SRBCT.mat',...
                'AR10P.mat',            'lymphoma.mat',      'RELATHE.mat',           'GLIOMA.mat',           'BASEHOCK.mat',...
                'Gisette.mat',          'Leukemia1',         'DLBCL.mat',             '9Tumor.mat',           'Brain1.mat',...
                'Prostate-GE.mat',      'CNS.mat',           'Leukemia.mat',          'ALLAML.mat',           'Carcinom.mat',...
                'nci9.mat',             'arcene.mat',        'orlraws10P.mat',        'Brain2.mat',           'CLL-SUB-111.mat',...
                '11Tumor.mat',          'SMK-CAN-187.mat',   'GLI-85.mat',            'Bhattacharjee-2001.mat','Gordon-2002.mat',...
                'Yeoh-2002-v1.mat'};
            
            Data = load(str{dataNo});
            obj.dataNo = dataNo;
            obj.boolTrain = 1;
            
            %% Shuffle the order of instances
            rng(1);   %Fix the random number seed
            randIndex = randperm(size(Data.X, 1));
            %save('GasSensor_index.mat', 'randIndex');

            Data.X = Data.X(randIndex',:);
            Data.Y = Data.Y(randIndex',:);

            %% Add noise to each feature
            noiseIndex = randperm(size(Data.X, 2), round(NoiseLevel./100.*size(Data.X, 2)));
            noiseVal   = rand(size(Data.X, 1), round(NoiseLevel./100.*size(Data.X, 2)));
            Data.X(1:size(Data.X, 1)*0.7, noiseIndex) = noiseVal(1:size(Data.X, 1)*0.7, :);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %% Normalize the input data
            Fmin         = min(Data.X, [], 1);
            Fmax         = max(Data.X, [], 1);
            Data.X       = (Data.X - repmat(Fmin, size(Data.X, 1), 1))./repmat(Fmax - Fmin, size(Data.X, 1), 1);
            obj.Category = unique(Data.Y);
            %% Divide the training set and test set
            obj.TrainIn  = Data.X(1:ceil(end*0.7), 1:end);
            obj.TrainOut = Data.Y(1:ceil(end*0.7), 1:end);
            obj.TestIn   = Data.X(ceil(end*0.7)+1:end, 1:end);
            obj.TestOut  = Data.Y(ceil(end*0.7)+1:end, 1:end);
            obj.K        = 5;   % The number of nearest neighbor in KNN  
            obj.TestNum  = 5;

            % Number of objectives and features
            obj.M = objectiveNo;
            obj.D = size(obj.TrainIn, 2) + size(obj.TrainIn, 1);
            switch encode
                case 1
                    obj.encoding = 'binary';
                case 2
                    obj.encoding = 'real';
                    obj.lower    = zeros(1, obj.D);
                    obj.upper    = ones( 1, obj.D);
            end

            [obj.FullErrorRate, obj.FullMinErrorRate] = CalFullErrorRate(obj);

            function [FullErrorRate, FullMinErrorRate] = CalFullErrorRate(obj)
                % The error ratio and the worst-class error ratio using all features and all instances
                indices     = crossvalind('Kfold', size(obj.TrainIn, 1), obj.TestNum);
                sumAccuracyRate= 0;
                eachClassAccuracyRate = zeros(size(obj.Category, 1), 1);
                for j = 1:obj.TestNum                % 5-fold cross validation
                    test        = (indices == j);    % Every iteration selects a fold as the test set 
                    train       =~ test;                
                    TestOutsub  = obj.TrainOut(test, :);

                    mdl = fitcknn(obj.TrainIn(train, :), obj.TrainOut(train, :), 'NumNeighbors', 5);
                    [Out, ~, ~] = predict(mdl, obj.TrainIn(test, :));
                    BalanceAccuracy = 0;
                    for t = 0:size(obj.Category, 1)-1
                        index = Out==t;
                        if sum(index) > 0
                            BalanceAccuracy = BalanceAccuracy + mean(Out(index)==TestOutsub(index));
                            eachClassAccuracyRate(t+1, 1) = eachClassAccuracyRate(t+1, 1) + mean(Out(index)==TestOutsub(index));
                        end
                    end
                    sumAccuracyRate   = sumAccuracyRate + BalanceAccuracy./size(obj.Category, 1);
                end
                AccuracyRate     = sumAccuracyRate./obj.TestNum;
                FullMinErrorRate = 1 - min(eachClassAccuracyRate./obj.TestNum);
                FullErrorRate    = 1 - AccuracyRate;
            end
            disp([obj.FullErrorRate, obj.FullMinErrorRate]);

        end
        %% Calculate objective values
        function PopObj = CalObj(obj, PopDec)
            [~, FeaNum] = size(obj.TrainIn);
            PopObj      = zeros(size(PopDec, 1), obj.M);
            obj.PopWorstError = zeros(size(PopDec, 1), 1);
            obj.PopError      = zeros(size(PopDec, 1), 1);
            if obj.boolTrain == 1   % To judge whether in the training set or the test set
                %%%%% For training set %%%%%
                theta = 0.5;     % The threshold to determine whether a feature is selected or discarded
                PopDec(PopDec < theta)  = 0 ;
                PopDec(PopDec >= theta) = 1;
                PopDec = logical(PopDec);
                boolFeaSel = PopDec(:, 1:FeaNum);
                boolInsSel = PopDec(:, FeaNum+1:end);
                for i = 1 : size(PopObj, 1)
                    if sum(PopDec(i, :)) == obj.D
                        PopObj(i, :) = [1, 1, obj.FullErrorRate];
                        obj.PopWorstError(i, 1) = obj.FullMinErrorRate;
                        obj.PopError(i, 1) = obj.FullErrorRate;
                    else
                    TrainInsub  = obj.TrainIn(boolInsSel(i, :)', :);
                    TrainOutsub = obj.TrainOut(boolInsSel(i, :)', :);
                    TestInsub   = obj.TrainIn(~boolInsSel(i, :)', :); 
                    TestOutsub  = obj.TrainOut(~boolInsSel(i, :)', :);
                    eachClassAccuracy = zeros(size(obj.Category, 1), 1); 

                    %disp([size(TrainInsub(:, boolFeaSel(i, :))), size(TrainOutsub)]);
                    mdl = fitcknn(TrainInsub(:, boolFeaSel(i, :)), TrainOutsub, 'NumNeighbors', 5);
                    [Out, ~, ~] = predict(mdl, TestInsub(:, boolFeaSel(i, :)));
                    for t = 0:size(obj.Category, 1)-1
                        index = Out==t;
                        if sum(index) > 0
                            eachClassAccuracy(t+1, 1) = mean(Out(index)==TestOutsub(index));
                        end
                    end
                    obj.PopWorstError(i, 1) = 1 - min(eachClassAccuracy);
                    obj.PopError(i, 1) = 1 - sum(eachClassAccuracy)./size(obj.Category, 1);
                    PopObj(i, 1) = mean(boolFeaSel(i, :)); % Selected feature ratio
                    PopObj(i, 2) = mean(boolInsSel(i, :)); % Selected instance ratio
                    PopObj(i, 3) = 1 - sum(eachClassAccuracy)./size(obj.Category, 1); % Classification error rate
                    end 
                end
            else
                %%%%% For test set %%%%%
                %PopDec = ones(size(PopDec,1), size(PopDec,2)); % Full set
                PopDec = logical(PopDec);
                boolFeaSel = PopDec(:, 1:FeaNum);
                boolInsSel = PopDec(:, FeaNum+1:end);
                for i = 1 : size(PopObj, 1)
                    mdl = fitcknn(obj.TrainIn(boolInsSel(i, :)', boolFeaSel(i, :)), obj.TrainOut(boolInsSel(i, :)', :), 'NumNeighbors', 5);
                    [Out, ~, ~] = predict(mdl, obj.TestIn(:, boolFeaSel(i, :)));

                    BalancedAccuracy = 0;
                    %errorList = ones(1, size(obj.Category, 1));
                    for t = 0:size(obj.Category, 1)-1
                        index = Out==t;
                        if sum(index) > 0
                            BalancedAccuracy = BalancedAccuracy + mean(Out(index)==obj.TestOut(index));
                            %errorList(1, t+1) = 1 - mean(Out(index)==obj.TestOut(index));
                        end
                    end
                    BalancedError = 1 - BalancedAccuracy./size(obj.Category, 1);
                    PopObj(i, 1) = mean(boolFeaSel(i, :)); % Selected feature ratio
                    PopObj(i, 2) = mean(boolInsSel(i, :)); % Selected instance ratio
                    PopObj(i, 3) = BalancedError;
                end 
            end
        end
        
        %% Calculate constraints
        function PopCon = CalCon(obj, PopDec)
            if obj.boolTrain == 1   % To judge whether in the training set or the test set
                PopCon(:,1) = obj.PopError - obj.FullErrorRate;
                PopCon(:,2) = obj.PopWorstError - obj.FullMinErrorRate;
            else
                PopCon = zeros(size(PopDec, 1), 2);
            end
        end
        
        %% Display a population in the objective space
        function DrawObj(obj, Population)
            %Draw(Population.objs, {'Selected feature ratio', 'Selected instance ratio', 'Classification error rate'});
            Draw(Population.objs, {'', '', 'Classification error rate'});
        end 
    end
end