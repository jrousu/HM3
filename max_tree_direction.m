function [Mu_1,ipos,ineg,izero,Gmax,Ymax] = max_tree_direction(gradient,mu,chunk,C,cut_off,tree_inference)

% find a feasible direction that maximizes the gradient for the tree
% the direction is chosen among vertices of the constraint set, 
% defined by the edge- box constraint and marginal consistency constraints

global E;
global tolerance;



if nargin < 5
    cut_off = tolerance;
end
if nargin < 6
    tree_inference = 0;
end
m = numel(gradient)/(4*size(E,1));
% find maximum gradient edge-labeling 
%[Gmax,Ymax] = max_gradient_labeling(gradient,tree_inference);
[Gmax,Ymax] = max_gradient_labeling(gradient,tree_inference);
Ymax = (2*(Ymax(:,E(:,1))-1) + Ymax(:,E(:,2)))';

% find positive and negative max gradient examples within the chunk
ipos = intersect(find(Gmax > cut_off),1:length(chunk));
ineg =  intersect(find(Gmax <= -cut_off),1:length(chunk)); 
% examples with close to zero max gradient and those outside the chunk 
izero = setdiff(1:length(chunk),union(ipos,ineg));

Mu_1 = zeros(size(E,1)*4,m); mu = reshape(mu,size(E,1)*4,m);
Ymax = Ymax(:,ipos);
for u = 1:4
    Mu_1(4*(1:size(E,1))-4 + u,ipos) = C*(Ymax == u); % direction of the min-margin (max-gradient) labeling for examples with large enough gradient
end
Mu_1(:,ineg) = 0; % examples with negative max gradient (positive min-margin) are updated towards zero
Mu_1(:,izero) = mu(:,izero); % the rest of the examples are left alone

Mu_1 = reshape(Mu_1,4,size(E,1)*m);


