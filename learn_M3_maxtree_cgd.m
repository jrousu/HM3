function [mu,Y_pred] = learn_M3_maxtree_cgd(params)

global E;
global m;
global X_tr;
global Y_tr;
global X_ts;
global Y_ts;

global Kx_tr;
global Kx_ts;

global verbosity;

global t_E;
global tolerance;
global epsilon;

global Kmu;
global Kmu_1;

global mu;

M3_set_globals(E,params);
start_time = cputime; next_profile_tm = start_time + params.profile_tm_interval;

mu = zeros(4,size(E,1)*m); 
WS = 1:m; 
C = params.slack;

chunksize = min([params.chunksize;params.ws_workmem_limit/(8*m);m]);
gradient = compute_loss_vector(params.hloss,params.scaling,params.mloss);
gradient = reshape(gradient,4,size(E,1)*m);
Kmu = zeros(size(gradient));


obj = 0; iter = 0;

if params.profiling
    print_message('Initial error rates...',3);
    t_E = reshape(compute_loss_vector(0,0,1) == 0,4,m*size(E,1));
    [n_err,p_err,n_err_microlbl,p_err_microlbl] = compute_M3_error(WS,mu,Y_tr,gradient + Kmu,E,1,1);
    [microlabel_errors_ts,err_ts,time,Y_pred] = test_M3(mu,E,verbosity);
    tm = cputime;
        print_message(sprintf('tm: %d err_tr: %d (%3.2f) ml.loss tr: %d (%3.2f) err_ts: %d (%3.2f) ml.loss ts: %d (%3.2f)',...
            round(tm-start_time),n_err,p_err,n_err_microlbl,p_err_microlbl*100,round(err_ts*size(Y_ts,1)),err_ts*100,sum(microlabel_errors_ts),sum(microlabel_errors_ts)/numel(Y_ts)*100),0,params.profile_log);
    print_message(sprintf('%d ',microlabel_errors_ts),4);
    t_E = []; Y_pred = [];
end

print_message('Starting descent...',0);

chunk = 1:m; 
target_update_size = m*size(E,1);

cut_off = 0; %params.epsilon; % gradients smaller than this will be considered zero

dgap = inf; primal_ub = inf; randomize = 0; tree_inference = 0;
while and(primal_ub - obj > params.epsilon*obj,iter < params.maxiter)

    print_message('Computing x-kernel...',3);
    get_x_kernel(chunk);

    print_message('Initial gradient...',3);

    Kmu = compute_Kmu(mu,Kx_tr,chunk);
    
    print_message('Min-margin labeling and duality gap...',3);        
    [duality_gap,primal_ub,xi,min_margin] = compute_duality_gap_vector(WS,gradient,obj,C,primal_ub);

    dgap_chunk = sum(duality_gap(chunk));

    print_message('Optimizing chunk...',3);
    chunk_iter = 0; failed = 0; success = 0; x_neg = 1:m; x_pos = []; x_zero = [];
    while and(sum(duality_gap(chunk)) > obj*params.epsilon , iter < params.maxiter)
        iter = iter + 1;
        print_message('Update direction...',4);
        tree_inference = ~tree_inference;
        x_neg_cur = x_neg; x_pos_cur = x_pos; x_zero_cur = x_zero;
        [d,x_pos,x_neg,x_zero,Gmax] = max_tree_direction(gradient,mu,chunk,C,cut_off,tree_inference);
        print_message(sprintf('p: %d z: %d n: %d dp: %d dz: %d dn: %d',...
            length(x_pos),length(x_zero),length(x_neg),length(setxor(x_pos_cur,x_pos)),length(setxor(x_zero_cur,x_zero)),length(setxor(x_neg_cur,x_neg))),3);

        print_message('Conditional gradient...',4);
        compute_Kmu_1(d,x_pos,x_zero,x_neg,chunk); Kmu_1 = Kmu_1 - Kmu;
        d = d - mu; 
        
        print_message('Step to saddle point...',4);

        l = gradient(:)'*d(:); q = d(:)'*(Kmu_1(:));
        alpha = min(l/q,1);

        delta_obj = l*alpha - alpha^2/2*q;

        update_size = floor(sum(sum(abs(d)) > tolerance)/(4*size(E,1)));
        % feasible alpha found
        if alpha > 0
            
            obj = obj + delta_obj; 
            
            mu = mu + alpha*d;
            gradient = gradient - alpha*Kmu_1;
            Kmu = Kmu + alpha*Kmu_1;
            success = success + 1; failed = 0;
            if and(success > 2,update_size < chunksize)
                % increase update size, if successful an update was small
                sGmax = flipud(sort(Gmax));
                cut_off = max(min(sGmax(chunksize) - tolerance,params.epsilon),0);
            end
        else
            success = 0; failed = failed + 1;
            break
        end

        print_message('Min-margin labeling and duality gap...',4);
        [duality_gap,primal_ub,xi,min_margin] = compute_duality_gap_vector(WS,gradient,obj,C,primal_ub);        

        chunk_iter = chunk_iter + 1;

        if verbosity == 4
            whos; whos('global');
        end

        tm = cputime;

        print_message(sprintf('alg: TCGD tm: %d  iter: %d obj: %f dgap: %f xi: %f chunkiter: %d chunksize: %d xi(chunk): %f ||d||1: %f #x_upd: %d alpha: %1.10f max_mu: %f cutoff: %f rand: %d fail: %d',round(tm-start_time),iter,obj,primal_ub-obj,sum(xi),...
            chunk_iter,chunksize,sum(xi(chunk)),max(max(abs(d))),update_size,alpha,max(max(mu)),cut_off,randomize,failed),2,params.profile_log);

        if params.profiling
            if tm > next_profile_tm
                next_profile_tm = next_profile_tm + params.profile_tm_interval;    
                t_E = reshape(compute_loss_vector(0,0,1) == 0,4,size(E,1)*m);
                [n_err,p_err,n_err_microlbl,p_err_microlbl] = compute_M3_error(WS,mu,Y_tr,gradient  + Kmu,E,1,1);
                [microlabel_errors_ts,err_ts,time,Y_pred] = test_M3(mu,E,verbosity);
                tm = cputime;
                print_message(sprintf('tm: %d err_tr: %d (%3.2f) ml.loss tr: %d (%3.2f) err_ts: %d (%3.2f) ml loss ts: %d (%3.2f)',...
                    round(tm-start_time),n_err,p_err*100,n_err_microlbl,p_err_microlbl*100,round(err_ts*size(Y_ts,1)),err_ts*100,sum(microlabel_errors_ts),sum(microlabel_errors_ts)/numel(Y_ts)*100),0,params.profile_log);
                print_message(sprintf('%d ',microlabel_errors_ts),0);
		sfile = sprintf('Ypred_%d%d.mat',params.hloss,params.scaling);
                save(sfile,'Y_pred');
                t_E = []; Y_pred = [];
            end
        end
    end
end

t_E = reshape(compute_loss_vector(0,0,1) == 0,4,size(E,1)*m);
[microlabel_errors_ts,err_ts,time,Y_pred] = test_M3(mu,E,verbosity);









