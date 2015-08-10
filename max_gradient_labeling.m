function  [Gmax,Ymax] = max_gradient_labeling_ls(G,tree_inference)

global E;

global LevelEdges;
global LevelNodes;
global M_Parent;
global Eparent;

if nargin < 2
    tree_inference = 0;
end

m = numel(G)/(4*size(E,1));

G11 = reshape(G(1,:),size(E,1),m)'; 
if ~tree_inference 
    G12 = reshape(G(2,:),size(E,1),m)'; 
end
G21 = reshape(G(3,:),size(E,1),m)'; G22 = reshape(G(4,:),size(E,1),m)'; 

% for  conditional labelings
Ymax = zeros(m,size(E,1)+1);
Ymax_e_1 = zeros(m,size(E,1));
Ymax_e_2 = zeros(m,size(E,1));

% make a breadth-first bottom-up pass over the hierarchy computing
% the maximum gradient of the subtree and storing in Ymax_e_u(x,e) = u
% for each edge the best label v for e(2) given e(1) is labeled u

for level = length(LevelEdges):-1:1
    
    edges = LevelEdges{level};
          
    % obtain maximum gradients for subtrees on level conditioned on parent
    % label
    if tree_inference
        Gmax1 = G11(:,edges);
        Ymax_e_1(:,edges) = 1; % maximum grad label for e(2) given parent -
    else
        Gmax1 = max(G11(:,edges),G12(:,edges)); % maximum gradient given parent -
        Ymax_e_1(:,edges) = (G12(:,edges)> G11(:,edges)) + 1; % maximum grad label for e(2) given parent -
    end
    
    Gmax2 = max(G21(:,edges),G22(:,edges)); % maximum given parent +
    Ymax_e_2(:,edges) = (G22(:,edges) > G21(:,edges)) + 1; % maximum label for e(2) given parent +
    
    % compute the sum of maximum gradients of subtrees rooted by edges
    % for the parent edges
    if level > 1
        parent_edges = LevelEdges{level-1};
        Smax = Gmax1*M_Parent(edges,parent_edges);
        G11(:,parent_edges) = G11(:,parent_edges) + Smax;
        G21(:,parent_edges) = G21(:,parent_edges) + Smax;
        
        Smax = Gmax2*M_Parent(edges,parent_edges);       
        if ~tree_inference
            G12(:,parent_edges) = G12(:,parent_edges) + Smax;
        end
        G22(:,parent_edges) = G22(:,parent_edges) + Smax;
    end
    
end

Gmax = max(Gmax1,Gmax2);

if nargout > 1

    % find the maximum gradient labeling by making a top-down breadth first
    % pass of the hierarchy by choosing the best label for e(2) given that
    % e(1) is fixed according to global maximum gradient

    Ymax(:,E(edges,1)) = (Gmax == Gmax2) + 1;
    Ymax_e = Ymax_e_1(:,edges).*(Ymax(:,E(edges,1)) == 1) + Ymax_e_2(:,edges).*(Ymax(:,E(edges,1)) == 2);

    children = find(sum(M_Parent(:,edges),2));
    while  ~isempty(children)
        
        Ymax(:,E(edges,2)) = Ymax_e;
        I_Children2 = Ymax_e*M_Parent(children,edges)' == 2;
        
        if tree_inference
            Ymax_e = ~I_Children2 + Ymax_e_2(:,children).*I_Children2;
        else
            Ymax_e = Ymax_e_1(:,children).*~I_Children2 + Ymax_e_2(:,children).*I_Children2;
        end
        
        edges = children;
        children = find(sum(M_Parent(:,children),2));        
    end        
    Ymax(:,E(edges,2)) = Ymax_e;
end














    
    
    
    
    
    
    
    











