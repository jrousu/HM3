function [mu_1,Gmax,Ymax,ipos,ineg,izero] = max_tree_direction(gradient,mu,chunk,C,cut_off,tree_inference)

% find a feasible direction that maximizes the gradient for the tree
% the direction is chosen among vertices of the constraint set, 
% defined by the edge- box constraint and marginal consistency constraints

global E;
global tolerance;
global m;


if nargin < 5
    cut_off = tolerance;
end
if nargin < 6
    tree_inference = 0;
end

% find maximum gradient edge-labeling 
[Gmax,Ymax] = max_gradient_labeling_x(gradient,tree_inference);
Ymax = 2*(Ymax(:,E(:,1))-1) + Ymax(:,E(:,2));

% convert tu marginal dual variable vector
mu_1 = zeros(size(mu));
if Gmax > cut_off
    for u = 1:4
        mu_1(4*(1:size(E,1))-4 + u) = C*(Ymax == u); 
    end
elseif Gmax > -cut_off
    mu_1 = mu;
end

