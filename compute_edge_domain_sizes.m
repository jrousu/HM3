function eDom = compute_edge_domain_sizes(E,yDom)

% computes the number of value combinations each edge can take as the
% product of the domain size of the variables tied to the edge

for e = E'
    if e(1) > 0
        eDom(e(3),1) = prod(yDom(e(1:2)',1));
    else
        eDom(e(3),1) = 2*yDom(e(2)); % assume binary labeling
    end
end

% compute the cumulative sum of the domain sizes 
% in the order edges are stored in the matrix E. 
eDom(1,2) = eDom(1,1);

for eindex = 2:size(E,1)
    eDom(eindex,2) = eDom(eindex-1,2) + eDom(eindex,1);
end

