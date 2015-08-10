function loss_scaling = compute_loss_scaling_new(scaling,mlloss)

global NodeLosses;
global NodeDegree;
global Treesize;
global E;
global l;
global root;

if nargin < 1
    scaling = 0;
end

NodeDegree = zeros(1,l);
for i = 1:l
    NodeDegree(i) = sum(sum(E(:,1:2) == i));
end

%root = 1;
NodeLosses(root) = 1;

switch scaling
    case 0
        NodeLosses = ones(1,l);
    case 1
    edges = E(E(:,1) == root,:);
    for edge = edges'
        compute_sibling_scaling(edge,size(edges,1));
    end
    case 2
        NodeLosses = Treesize/l;
end

loss_scaling = NodeLosses;

if mlloss 
    loss_scaling = NodeLosses./NodeDegree; % divide the node loss to adjacent edges
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function compute_sibling_scaling(edge,no_of_siblings)

global E;
global NodeLosses;

NodeLosses(edge(2)) = NodeLosses(edge(1))/no_of_siblings;

edges = E(E(:,1) == edge(2),:);
for edge = edges'
    compute_sibling_scaling(edge,size(edges,1));
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function compute_document_scaling(edge)

global E;
global NodeLosses;
global DocumentFreq;

NodeLosses(edge(2)) = DocumentFreq(edge(2))/DocumentFreq(edge(1));

edges = E(E(:,1) == edge(2),:);
for edge = edges'
    compute_document_scaling(edge);
end 


