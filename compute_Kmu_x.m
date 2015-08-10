function [Kmu_x] = compute_Kmu_x(x,mu,Kx)

global m;
global E;

global IndEdgeVal;

global Rmu;
global Smu;
global term12;
global term3;
global term4;

for u = 1:4

    Ind_te_u = full(IndEdgeVal{u}(:,x));   
    H_u = Smu{u}*Kx-Rmu{u}*Kx;
    
    term12(1,Ind_te_u) = H_u(Ind_te_u)';
    term34(u,:) = -H_u';
end

Kmu_x = reshape(term12(ones(4,1),:) + term34,4*size(E,1),1);








