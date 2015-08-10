function  [Gmax,Ymax] = max_gradient_labeling_x(gradient,tree_inference)

global E;

global LevelEdges;
global LevelNodes;
global M_Parent;
global Eparent;
global Echildren;

if nargin < 2
    tree_inference = 0;
end

m = length(gradient)/(4*size(E,1));

% reorganize gradient to four matrices, one for each potential
% edge-labeling
G = reshape(gradient,4,m*size(E,1)); % to find max subtree labeling, restrict to positive gradient

G11 = reshape(G(1,:),size(E,1),m)'; 
if ~tree_inference 
    G12 = reshape(G(2,:),size(E,1),m)'; 
end
G21 = reshape(G(3,:),size(E,1),m)'; G22 = reshape(G(4,:),size(E,1),m)'; 

% for sums of subtree gradients
Smax1 = zeros(m,size(E,1));
Smax2 = zeros(m,size(E,1));

% for labelings and conditional labelings
Ymax_e = zeros(m,size(E,1));
Ymax_e_1 = zeros(m,size(E,1));
Ymax_e_2 = zeros(m,size(E,1));

% make a breadth-first bottom-up pass over the hierarchy computing
% the maximum gradient of the subtree and storing in Ymax_e_u(x,e) = u
% for each edge the best label v for e(2) given e(1) is labeled u

for level = length(LevelEdges):-1:1
    
    edges = LevelEdges{level};
    
    % add the maximum gradients of the subtrees hanging from the edges
    G11(:,edges) = G11(:,edges) + Smax1(:,edges);
    G21(:,edges) = G21(:,edges) + Smax1(:,edges);
    if ~tree_inference
        G12(:,edges) = G12(:,edges) + Smax2(:,edges);    
    end
    G22(:,edges) = G22(:,edges) + Smax2(:,edges);
      
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
        Smax1(:,parent_edges) = Gmax1*M_Parent(edges,parent_edges);
        Smax2(:,parent_edges) = Gmax2*M_Parent(edges,parent_edges);
    end
    
end

Gmax = max(Gmax1,Gmax2);

if nargout > 1

    % find the maximum gradient labeling by making a top-down breadth first
    % pass of the hierarchy by choosing the best label for e(2) given that
    % e(1) is fixed according to global maximum gradient

    Ymax(:,E(edges,1)) = (Gmax == Gmax2) + 1;
    Ymax_e(:,edges) = Ymax_e_1(:,edges).*(Ymax(:,E(edges,1)) == 1) + Ymax_e_2(:,edges).*(Ymax(:,E(edges,1)) == 2);


    for level = 1:(length(LevelEdges)-1);

        edges = LevelEdges{level};
        children = LevelEdges{level+1};
    
        if tree_inference
          Ymax_e(:,children) = (Ymax_e(:,edges) == 1)*M_Parent(children,edges)'...
              + Ymax_e_2(:,children).*((Ymax_e(:,edges) == 2)*M_Parent(children,edges)');
        else
            Ymax_e(:,children) = Ymax_e_1(:,children);
            edges2 = Ymax_e(edges) == 2;
            children2 = Ymax_e(edges)*M_Parent(children,edges)' == 2;
            Ymax_e(children(children2)) = Ymax_e_2(:,children(children2));
        end

    end

    Ymax(:,E(:,2)) = Ymax_e;

end














    
    
    
    
    
    
    
    











