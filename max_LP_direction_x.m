function [mu_1,Gmax,Ymax,ipos,ineg,izero] = max_LP_direction_x(gradient,mu_x,chunk,C,cut_off,tree_inference)

% find a feasible direction that maximizes the gradient for the tree
% the direction is chosen among vertices of the constraint set, 
% defined by the edge- box constraint and marginal consistency constraints

global E;
global tolerance;
global m;
global linprogoptions;

global Aeq_x;
global A_x;
global b;
global beq;
global lb;
global ub;


%[mu_1,fval,exitflag,output] = linprog(-gradient, A_x,b,Aeq_x,beq,lb,ub,mu_x,linprogoptions); 
defaultopt = struct('Display','final',...
   'TolFun',1e-8,'Diagnostics','off',...
   'LargeScale','on','MaxIter',85, ...
   'Simplex','off');

[mu_1,fval,lambda,exitflag,output] = lipsol(-gradient,A_x,b,Aeq_x,beq,lb,ub,linprogoptions,defaultopt,0);
Gmax = gradient'*mu_1;

% compute the (potentially fractional) labeling corresponding to mu_1

Mu_1 = reshape(mu_1,4,size(E,1));

% Y_max(e) = (mu_e(i,1,0) +  mu_e(i,1,1))/C
Y_max(E(:,1)') = sum(Mu_1(3:4,:))/C;
Y_max(E(:,2)') = sum(Mu_1([2,4],:))/C;
