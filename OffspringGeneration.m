function Offspring = OffspringGeneration(Parent)
    %% Offspring generation of MFIs
    [proC, proM] = deal(1, 1);
    Problem = PROBLEM.Current();
    [InsNum, FeaNum] = size(Problem.TrainIn);
    
    if isa(Parent(1), 'SOLUTION')
        Parent = Parent.decs;
    end

    Parent1 = Parent(1:floor(end/2), :);
    Parent2 = Parent(floor(end/2) + 1:floor(end/2)*2, :);
    [N, D]  = size(Parent1);
    
    %% Genetic operators for binary encoding
    % One point crossover
    k = repmat(1:D, N, 1) > repmat(randi(D, N, 1), 1, D);
    k(repmat(rand(N,1) > proC, 1, D)) = false;
    Offspring1    = Parent1;
    Offspring2    = Parent2;
    Offspring1(k) = Parent2(k);
    Offspring2(k) = Parent1(k);
    Offspring     = [Offspring1; Offspring2];
    % Bit-flip mutation
    Site = rand(2*N, D) < proM/D;
    Offspring(Site) = ~Offspring(Site);

    % repair duplicated feature subsolutions
    boolis = ismember(Offspring(:, :), Parent(:, :), 'rows');
    normal = Offspring(boolis==0, 1:end);
    duplic = Offspring(boolis==1, 1:end);
    for i =1:size(duplic, 1)
        index1 = randi(FeaNum, 1);
        index2 = randi(InsNum, 1);
        duplic(i, index1) = ~duplic(i, index1);
        duplic(i, index2+FeaNum) = ~duplic(i, index2+FeaNum);
    end
    Offspring = [normal; duplic];

    % get unique offspring (function evaluated)
    for i=1:size(Offspring, 1)-1
        boolis = ismember(Offspring(i, :), Offspring(i+1:end, :), 'rows');
        if boolis==1
            index1 = randi(FeaNum, 1);
            index2 = randi(InsNum, 1);
            Offspring(i, index1) = ~Offspring(i, index1);
            Offspring(i, index2+FeaNum) = ~Offspring(i, index2+FeaNum);
        end
    end

    if Problem.M == 3
        % Repair solutions that do not select any instance for the given class
        ClassCategory = unique(Problem.TrainOut);
        PopIns = Offspring(:, FeaNum+1:end);
        for t=1:size(ClassCategory, 1)
            loc = find(Problem.TrainOut==ClassCategory(t, :));
            flag2 = sum(PopIns(:, loc'), 2) == 0;
            if sum(flag2, 1) > 0
                PopIns(flag2, loc') = randi([0, 1], sum(flag2, 1), size(loc, 1));
            end
        end
        Offspring(:, FeaNum+1:end) = PopIns;

        % Repair solutions that do not select any features
        PopFea = Offspring(:, 1:FeaNum);
        while sum(sum(PopFea, 2) ==0, 1) > 0   % This is facilate to 5-fold cross validation
            flag = sum(PopFea, 2)==0;
            PopFea(flag, :) = randi([0, 1], sum(flag, 1), FeaNum);
        end
        Offspring(:, 1:FeaNum) = PopFea;
    end
   
    Offspring = SOLUTION(Offspring);
end
