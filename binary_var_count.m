function count = binary_var_count

% counts the number of binary marginal dual variables

global m;
global eDom;

count = m*eDom(end,2);
