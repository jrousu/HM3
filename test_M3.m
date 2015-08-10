function [microlabel_errors,err,time,Y_pred] = test_M3(mu,E,verbosity)    

global ws_workmem_limit;
global tree_inference;
global t_E;

global X_ts;
global X_tr;
global Y_ts;
global Kx_ts;

global x_datasource;


%start the clock
tmstart = cputime;

if x_datasource == 1
    m = size(X_tr,1);
    m_ts = size(X_ts,1);
else
    m = size(Kx_ts,1);
    m_ts = size(Kx_ts,2);
end


ws_chunksize = min(floor(ws_workmem_limit/(8*m)),m_ts);

last_x = 0;
while last_x < m_ts
    first_x = last_x +1; last_x = min(last_x + ws_chunksize,m_ts); chunk = first_x:last_x;
    w_phi_e = compute_w_phi_e(Kx_ts);
    
    m_ts = size(Kx_ts,2);
    [maxLikelihood,Y_pred(chunk,:)] = max_gradient_labeling(w_phi_e,tree_inference);
end


tmstop = cputime;
time = (tmstop - tmstart);

 if verbosity > 2
            fprintf('\n');
 end

microlabel_errors = sum(abs(Y_pred - Y_ts));

[i_err,j_err] = find(Y_pred - Y_ts);
err = length(unique(i_err))/m_ts;
