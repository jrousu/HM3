function tree_size = compute_tree_size(node)

global E;
global Treesize;

% find the edges starting from the node
edges = E(find(E(:,1) == node),:);
tree_size = 0;

% visit the subtrees recursively
for edge = edges'   
    tree_size = tree_size + compute_tree_size(edge(2));    
end 
tree_size = tree_size + 1;

Treesize(node) = tree_size;
