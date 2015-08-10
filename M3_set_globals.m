function M3_set_globals(Edges,params)

global Ind;

global l; % number of microlabels
global m; % number of training examples
global root;

global X_tr;
global X_ts;
global Y_tr;
global Y_ts;

global Kx_tr;
global Kx_ts;


global EdgeIndex; % an index for retrieving the indices of edges in the edge list
global EdgeList;
global Echildren; % children of an edge, used in inference
global Nchildren;
global Eparent;
global SE_parent;
global Iseparent;
global LevelEdges;


global M_Parent; % matrix with M_Parent(i,j) = 1 iff edge i is the child of edge j;

global uParents;
global NodeChildren;
global NodeParent;

global EdgeValues;
global IndEdgeVal;

global Treesize;

global yDom; % yDom(i,1) is the domain size of variable y_i, yDom(y,2) = sum_{j<=i} yDom(j,1)
global eDom; % eDom(e,1) and eDom(e,2) similarly for edges

global verbosity;
global profile_tm_interval;
global profiling;
global profile_log;
global output_tm_interval;

global next_output_tm;
global next_profile_tm;

global linprogoptions;
global lipsol_defaultopt;

global indices_EX;
global indices_XE;

global consistent_vars;   % dual variables that are consistent with a tree
global inconsistent_vars; 

global obj_ub;
global max_obj_WS;

global tolerance;
global epsilon;
global chunksize;
global normalize;
global enforce_tree;
global tree_inference;

% globals for compute_Kmu_x
global Rmu;
global Smu;
global Smu_u;
global term12;
global Mu;
global Mu_2;
global term_3;
global term4;

global chunksize;

global C;
global maxiter;

global ws_workmem_limit;

global E;

global max_IP_iter;
global x_datasource;


x_datasource = params.x_datasource;

E = Edges;

obj_ub = Inf;
max_obj_WS = Inf;



l = size(Y_tr,2);

if x_datasource == 1
    m = size(X_tr,1);
elseif x_datasource == 2
   error(sprintf('x_datasource == 2 not implemented'));
else
   m = size(Kx_tr,1);
end

EdgeList = Edges;
%EdgeIndex = generate_Edge_Index(EdgeList);
[yDom, eDom] = compute_domain_sizes(Y_tr,EdgeList);


yDom(find(yDom(:,1) < 2),1) = 2;


if (length(find(yDom(:,1) == 2)) == size(yDom,1)) % if binary y's
    [Ind,Ind2,Ind3] = precomputeIndicatorMatrices(4);
end

[Echildren,Eparent,M_Parent,SE_parent,Iseparent] = find_edge_neighbours(Edges);
uParents = unique(Eparent); uParents = uParents(uParents > 0);
root = E(find(Eparent > length(Eparent)),1);
LevelEdges = cell(1,max(E(:,4)));
for level = 1:length(LevelEdges)
    LevelEdges{level} = E(E(:,4) == level,3);
end

% Edge_offset = (1:size(E,1))*4-4;
% X_offset = [(1:m)*(Edge_offset(end)+4)-Edge_offset(end)-4]';
% e_indices =  1:(4*size(E,1));
% V_offset = X_offset(:,ones(1,4*size(E,1))) + e_indices(ones(m,1),:);

% compute the subtree sizes to matrix Treesize;
compute_tree_size(root);


% edge values of training opints
EdgeValues = zeros(m,size(E,1));

for e = E'
    if e(1) > 0
        EdgeValues(:,e(3)) = Y_tr(:,e(2)) + (Y_tr(:,e(1))-1)*yDom(e(2),1);
    else
        EdgeValues(:,e(3)) = Y_tr(:,e(2)) + yDom(e(2),1); % force node 0 as positive
    end
end

for u = 1:4
    IndEdgeVal{u} = sparse(EdgeValues == u)';
end
EdgeValues = [];


warning off MATLAB:divideByZero
lipsol_defaultopt = struct('Display','final',...
   'TolFun',[],'Diagnostics','off',...
   'LargeScale','on','MaxIter',[], ...
   'Simplex','off');
%linprogoptions = linprog('defaults');
linprogoptions.LargeScale = 'on';
linprogoptions.ShowStatusWindow = 'off';
linprogoptions.Simplex = 'off';
linprogoptions.Display = 'off';
linprogoptions.MaxIter = params.max_ip_iter; %eDom(end,2); % a heuristic limit: one iter. per dual var.
max_IP_iter = params.max_ip_iter;

profiling = params.profiling;
profile_tm_interval = params.profile_tm_interval;
profile_log = params.profile_log;
%output_tm_interval = params.output_tm_interval;

next_output_tm = 0;
next_profile_tm = 0;

verbosity = params.verbosity;
%normalize = params.normalize;
tolerance = params.tolerance;
epsilon = params.epsilon;
chunksize = params.chunksize;
enforce_tree = params.enforce_tree;
tree_inference = or(enforce_tree,params.tree_inference);

chunksize = min(params.chunksize,m);

C = params.slack;
maxiter = params.maxiter;

% indices_XE = permute_example_to_edge(1:binary_var_count,E);
% indices_EX = permute_edge_to_example(1:binary_var_count,E);

if enforce_tree
    inconsistent_vars = eDom(:,2)-eDom(:,1)+2;
    consistent_vars = setdiff(1:eDom(end,2),inconsistent_vars);
else
    inconsistent_vars = [];
    consistent_vars = 1:eDom(end,2);
end

ws_workmem_limit = params.ws_workmem_limit;

