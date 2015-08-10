function [duality_gap,primal_ub,xi,min_margin] = compute_duality_gap(WS,g,obj,C,primal_ub)

global mu;
global E;
global m;

Gmax = max_gradient_labeling(g);
min_margin = -Gmax;

xi = max(-min_margin,0);

duality_gap = C*xi - sum(reshape(sum(g.*mu),size(E,1),m))';

primal_ub = min(obj+sum(duality_gap),primal_ub);


