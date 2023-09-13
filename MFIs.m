classdef MFIs < ALGORITHM
% <multi> <real/binary/permutation> <constrained/none>
% Simultaneous Feature and Instance Selection via Evolutionary Multiform Search
% 7,11,13,18,23,25,27,29,33,35,36,37,40,15,28,42,50,26,41,51,30
% 30,27,7,11,23,25,40,35,29,37,50,36,33,18,42,41,13,51,28,15,26

%------------------------------- Reference --------------------------------
% Simultaneous Feature and Instance Selection via Evolutionary Multiform Search
%------------------------------- Copyright --------------------------------
% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm, Problem)
            %% Setting population size and maxFE for fair comparison %%
            [Problem.N, Problem.maxFE] = InitialExperimentSetting(Problem);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            
            %% Generate random population
            TargetPop = InitializePopulation(Problem);
            SourcePop = TargetPop;   
            initialE  = max(max(0, TargetPop.cons), [], 1);

            [TargetPop, FrontNoTP, ErrorRateTP] = EnvironmentalSelection(TargetPop, Problem.N, zeros(1,2));
            [SourcePop, FrontNoSP, ~          ] = EnvironmentalSelection(SourcePop, Problem.N, initialE);
            
         
            %% Optimization
            while Algorithm.NotTerminated(TargetPop)
                epsn       = ReduceBoundary(initialE, ceil(Problem.FE/Problem.N), ceil(Problem.maxFE/Problem.N)-1);
                MatingPoolTP = TournamentSelection(2, Problem.N/2, ErrorRateTP, FrontNoTP);
                MatingPoolSP = TournamentSelectionSP(2, Problem.N/2, FrontNoSP, SourcePop, TargetPop(MatingPool1));
                Offspring  = OffspringGeneration([TargetPop(MatingPoolTP), SourcePop(MatingPoolSP)]);
                [TargetPop, FrontNoTP, ErrorRateTP] = EnvironmentalSelection([TargetPop, Offspring], Problem.N, zeros(1,2));
                [SourcePop, FrontNoSP, ~          ] = EnvironmentalSelection([SourcePop, Offspring], Problem.N, epsn);

                %%%%% Applied to the test set %%%%%
                TargetPop = FSTraining2Test(Problem, TargetPop);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
        end
    end
end


function selectedIdx = TournamentSelectionSP(K, N, FrontNo, Pop, MatingPop)
    % K tournament selection for the source task
    PopDec = Pop.decs;
    MatingPopDec = MatingPop.decs;
    populationSize = length(FrontNo);
    selectedIdx = zeros(1, N);
    for i = 1:N
        % Randomly select two individuals
        idx = randperm(populationSize, K);      
        % Compare their fitness values
        if FrontNo(idx(1)) > FrontNo(idx(2))
            selectedIdx(i) = idx(2);
        elseif FrontNo(idx(1)) < FrontNo(idx(2))
            selectedIdx(i) = idx(1);
        else
            dis1 = pdist2(PopDec(idx(1), :), MatingPopDec(i, :), "hamming");
            dis2 = pdist2(PopDec(idx(2), :), MatingPopDec(i, :), "hamming");
            selectedIdx(i) = idx(TournamentSelection(2, 1, [-dis1, -dis2]));
        end
    end
end



function epsn = ReduceBoundary(eF, k, MaxK)
    %% Reduce the epsilon constraint boundary for source task
    z        = 1e-8;
    Nearzero = 1e-15;
    B        = MaxK./power(log((eF + z)./z), 1.0./10);
    B(B==0)  = B(B==0) + Nearzero;
    f        = eF.* exp( -(k./B).^10 );
    tmp      = find(abs(f-z) < Nearzero);
    f(tmp)   = f(tmp).*0 + z;
    epsn     = f - z;
    epsn(epsn<=0) = 0;
end

function Population = InitializePopulation(Problem)
    %% Initialize an population
    [InsNum, FeaNum] = size(Problem.TrainIn);
    Pop = zeros(Problem.N, Problem.D);
    ClassCategory = unique(Problem.TrainOut);
    ClassNum = size(ClassCategory, 1);

    lhs_samples = lhsdesign(Problem.N, 2);
    fea_range = [1/FeaNum, 0.9];
    ins_range = [ClassNum/InsNum, 0.9];

    num = [fea_range(1) + lhs_samples(:, 1)*(fea_range(2)-fea_range(1)), ins_range(1) + lhs_samples(:, 2)*(ins_range(2)-ins_range(1))];

    % Initialize subsolutions for selected features
    for i = 1 : Problem.N
        j = randperm(FeaNum, round(num(i, 1)*FeaNum));
        Pop(i, j) = 1;
    end

    % Initialize subsolutions for selected instances
    ClassInsNum = zeros(ClassNum, 1);
    ClassIns = cell(ClassNum, 1);
    SelClassInsNum = cell(ClassNum, 1);
    for i=1:ClassNum
        ClassInsNum(i, :) = sum(Problem.TrainOut==ClassCategory(i, :));
        ClassIns{i} = find(Problem.TrainOut==ClassCategory(i, :));
    end

    for i = 1 : Problem.N
        selnum = round(num(i, 2)*InsNum);
        for j=1:ClassNum
            SelClassInsNum{j} = zeros(size(ClassIns{j}));
        end
        sumNum = 0;
        while(sumNum < selnum)
            for j=1:ClassNum
                if sum(SelClassInsNum{j}) < ClassInsNum(j, :)
                    k = randperm(ClassInsNum(j, :)-sum(SelClassInsNum{j}), 1);
                    loc = find(SelClassInsNum{j}==0);
                    SelClassInsNum{j}(loc(k), 1) = 1;
                    Pop(i, FeaNum+ClassIns{j}(loc(k), 1)) = 1;
                    sumNum = sumNum + 1;
                    if sumNum >= selnum
                        break;
                    end
                end
            end
        end
    end
    Population = SOLUTION(Pop);
end
