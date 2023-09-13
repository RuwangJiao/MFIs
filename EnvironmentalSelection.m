function [Population, FrontNo, ErrorRate] = EnvironmentalSelection(Population, N, epsn)
% The environmental selection of MFIs

%------------------------------- Copyright --------------------------------
% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    CV = sum(max(0, Population.cons-epsn), 2);
    if sum(CV==0) > N
        %% Selection among feasible solutions
        Population = Population(CV==0);
        %% Non-dominated sorting
        [FrontNo, MaxFNo] = NDSort(Population.objs, N);
        Next = FrontNo < MaxFNo;
    
        %% Calculate the crowding distance of each solution
        CrowdDis = CrowdingDistance(Population.objs, FrontNo);
    
        %% Select the solutions in the last front based on their crowding distances
        Last     = find(FrontNo==MaxFNo);
        [~,Rank] = sort(CrowdDis(Last),'descend');
        Next(Last(Rank(1:N-sum(Next)))) = true;

        %% Population for next generation
        Population = Population(Next);
        FrontNo    = FrontNo(Next);
        %CrowdDis   = CrowdDis(Next);
    else
        %% Selection including infeasible solutions
        [~, rank]   = sort(CV);
        Population = Population(rank(1:N));
        [~, rnk, idxrnk] = unique(CV(rank(1:N)), 'stable');
        FrontNo = rnk(idxrnk)';
        %CrowdDis = zeros(1, N);
    end
    Obj = Population.objs;
    ErrorRate = Obj(:, 3)';
end