function [n_err,p_err,n_err_microlbl,p_err_microlbl,microlabel_errors] = compute_M3_error_WS(WS,mu,Y,delta_t,E,example_blocking,tree_inference) 

global X_tr;
global m;

global t_E;
global ws_workmem_limit;
global normalize;
global indices_EX;
global x_datasource;

global Ypred;

global Kx_tr;


if isempty(Ypred)
    Ypred = zeros(size(Y));
end

if nargin < 7
    tree_inference = 0;
end

w_phi_e = compute_w_phi_e( Kx_tr);
[maxLikelihood,Ypred] = max_gradient_labeling(w_phi_e,tree_inference);

microlabel_errors = sum(abs(Ypred-Y),2);
    
n_err_microlbl = sum(microlabel_errors); p_err_microlbl = n_err_microlbl/numel(Y);
n_err = sum(microlabel_errors > 0); p_err = n_err/length(microlabel_errors);
