function [yDom,eDom] = compute_domain_sizes(Y,EdgeList)

% computes two matrices
% yDom(i,1) is the domain size of variable i
% yDom(i,2) is the cumulative sum sum_{j<=i} yDom(i,1)

% eDom(eindex,1) is the 'domain size' of edge eindex, i.e. the number of
% unique value combinations, eDom(eindex,2) is the cumulative sum 
% sum_{findex<=eindex} eDom(findex,1)

yDom(:,1) = max(Y)'; % assuming that y values take a continuous range from 1 to max.

yDom(find(yDom(:,1) < 2),1) = 2;

yDom(1,2) = yDom(1,1);
for y = 2:size(yDom,1);
    yDom(y,2) = yDom(y-1,2) + yDom(y,1);
end

eDom = compute_edge_domain_sizes(EdgeList,yDom);
