function [Echildren,Eparent,M_Parent,SE_parent,Iseparent] = find_edge_neighbours(E)

Eparent = zeros(1,size(E,1));
M_Parent = zeros(size(E,1));
Nchildren = zeros(1,size(E,1)+1);
for e_index = 1:size(E,1);
    parentnode = E(e_index,1);
    childnode =  E(e_index,2);
    children = E(E(:,1) == childnode,3);
    Echildren{e_index} = children;
    Eparent(children) = E(e_index,3);
    M_Parent(children,E(e_index,3)) = 1;
end

[SE_parent,Iseparent] = sort(Eparent(Eparent > 0)');
Eparent(Eparent == 0) = size(E,1) + 1; % dummy parent



