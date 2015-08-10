function learn_M3_LP_cgd_decomp(params)

global eDom;
global start_time;
global m;
global IndEdgeVal;
global ws_workmem_limit;
global maxiter;
global consistent_vars;

global profiling;
global profile_tm_interval;
global next_profile_tm;
global indices_XE;
global verbosity;
global profile_log;

global tolerance;

global Kx_tr;
global Kx_ts;

global X_tr;
global Y_tr;
global X_ts;
global Y_ts;
global E;

global loss;
global mu;
global x_vars;

global Smu;
global Rmu;
global Kmu;

global A_x;
global b;
global Aeq_x;
global beq;
global lb;
global ub;

global t_E;

global C;


M3_set_globals(E,params);

Rmu = cell(1,4);
Smu = cell(1,4);
for u = 1:4
    Smu{u} = zeros(size(E,1),m);
    Rmu{u} = zeros(size(E,1),m);
end

start_time = cputime; next_profile_tm = start_time;
n_err = 0; p_err = 0; n_err_microlbl = 0; p_err_microlbl = 0; microlabel_errors = [];

print_message('Computing loss vector...',0);
loss = compute_loss_vector(params.hloss,params.scaling,params.mloss);
t_E = compute_loss_vector(0,0,1) == 0;
t_E = reshape(t_E,4*size(E,1),m);

print_message('Constraints...',0);
[A_x,b] = form_box_constraints_x(C);
[Aeq_x,beq] = form_consistency_constraints(E);
lb = zeros(size(A_x,2),1); ub = ones(size(A_x,2),1)*C;

Kxx_mu_x = zeros(4*size(E,1),m);
mu = zeros(4*size(E,1),m);

whos('Kx','mu','loss','E','Kd_X','Aeq_x','A_x','D_x''beq_x','b_x');

Kmu = zeros(binary_var_count,1);
gradient = loss;
obj = 0;
primal_ub = Inf;

WS = 1:m;

x_iter = zeros(m,1);
duality_gap_limit = params.epsilon*m/max(length(WS),1);
ws_chunksize_limit = min(floor(ws_workmem_limit/(8*m)));
x_blocksize = eDom(end,2);
d_x = zeros(x_blocksize,1);

edge_count = size(E,1);

print_message('Min-margin labeling and duality gap...',3);
gradient = reshape(gradient,4,size(E,1)*m); mu = reshape(mu,4,size(E,1)*m);
[duality_gap,primal_ub,xi,min_margin] = compute_duality_gap_vector(WS,gradient,obj,C,primal_ub);

iter = 0; opt_round = 1;

tm = cputime;

if params.profiling
    print_message('Initial error rates...',3);
    t_E = reshape(t_E,4,size(E,1)*m);
    mu = reshape(mu,4,size(E,1)*m);
    [n_err,p_err,n_err_microlbl,p_err_microlbl] = compute_M3_error(WS,mu,Y_tr,loss,E,1,1);
    [microlabel_errors_ts,err_ts,time,Y_pred] = test_M3(mu,E,verbosity);
    tm = cputime;
        print_message(sprintf('tm: %d err_tr: %d (%3.2f) ml.loss tr: %d (%3.2f) err_ts: %d (%3.2f) ml.loss ts: %d (%3.2f)',...
            round(tm-start_time),n_err,p_err,n_err_microlbl,p_err_microlbl*100,round(err_ts*size(Y_ts,1)),err_ts*100,sum(microlabel_errors_ts),sum(microlabel_errors_ts)/numel(Y_ts)*100),0,params.profile_log);
    print_message(sprintf('%d ',microlabel_errors_ts),4);
    Y_pred = [];  t_E = reshape(t_E,4*size(E,1),m); 
end

updated = zeros(m,1); skipped = zeros(m,1);

print_message('Starting descent...',0);

% repeat until working set converged and close to optima
chunk_last = 0; iter = 0; obj = 0;
while and(primal_ub - obj > params.epsilon*obj,iter < maxiter)

    mu = reshape(mu,4*size(E,1),m); loss = reshape(loss,4*size(E,1),m);
    
    x_iter(:) = 0;

    chunk_first = chunk_last + 1;
    chunk_last = chunk_last + min(ws_chunksize_limit,m);
    Chunk = mod((chunk_first:chunk_last)-1,m)+1;
    chunk_last = Chunk(end);

   get_x_kernel(Chunk);

    for i = 1:length(Chunk)

        x = Chunk(i);
        x_vars = (x-1)*x_blocksize + consistent_vars;
        loss_x = loss(:,x);
        mu_x = mu(:,x);
        te_x = t_E(:,x);

        % initial gradient
        Kmu_x = compute_Kmu_x(x,mu, Kx_tr(:,i));

        g_x =  loss_x - Kmu_x;

        % optimize x
        [mu_x,Kxx_mu_x(:,x),obj,x_iter(x)] = optimize_x(x,obj,mu_x,Kmu_x,Kxx_mu_x(:,x),loss_x,te_x,C,params.max_CGD_iter);

        % store sums of marginal dual variables, distributed by the
        % true edge values into Smu
        % store marginal dual variables, distributed by the
        % pseudo edge values into Rmu
        mu_x = reshape(mu_x,4,size(E,1));
        Smu_x = sum(mu_x)';
        for u = 1:4
            %Ind_te_u = find(EdgeValues(x,:) == u);
            te_u = IndEdgeVal{u}(:,x);
            Smu{u}(:,x) = Smu_x.*te_u;
            %Smu{u}(Ind_te_u,x) = Smu_x(Ind_te_u);
            Rmu{u}(:,x) = mu_x(u,:)';
        end
        mu(:,x) = reshape(mu_x,4*size(E,1),1);
%          obj1 = sum(sum(loss.*mu)) - 1/2*sum(sum(reshape(mu,4,size(E,1)*m).*compute_Kmu_b(reshape(mu,4,size(E,1)*m),Kx_tr,1:m,1:m)));
%          if abs(obj-obj1) > 0.1 
%              obj1;
%          end;
        iter = iter + x_iter(x);
        tm = cputime;
	%fprintf('alg: TCGDx tm: %d  x: %d iter: %d obj: %f dgap: %f xi: %f\n',...
        %    round(tm-start_time),x,iter,obj,primal_ub-obj,sum(xi));
        print_message(sprintf('alg: TCGDx tm: %d  x: %d iter: %d obj: %f dgap: %f xi: %f',...
            round(tm-start_time),x,iter,obj,primal_ub-obj,sum(xi)),4,params.profile_log);
    end
    
    mu = reshape(mu,4,size(E,1)*m);  loss = reshape(loss,4,size(E,1)*m); 
    max_chunk = 0; Kmu = zeros(4*size(E,1),m);
    print_message('Current full gradient...',3);
    while max_chunk < m
    	min_chunk = max_chunk +1;
    	max_chunk = min(min_chunk + params.chunksize -1,m); chunk = min_chunk:max_chunk;
    	Kmu(:,chunk) = reshape(compute_Kmu_b(mu,Kx_tr,chunk),4*size(E,1),length(chunk));
    end
    Kmu = reshape(Kmu,4,m*size(E,1),1);
    %disp(size(loss)); disp(size(Kmu));

    print_message('Min-margin labeling and duality gap...',3);
    tm = cputime;
    [duality_gap,primal_ub,xi,min_margin] = compute_duality_gap_vector(WS,loss-Kmu,obj,C,primal_ub);
    print_message(sprintf('alg: TCGDx tm: %d  iter: %d obj: %f mu: max %f min %f dgap: %f xi: %f',...
        round(tm-start_time),iter,obj,max(max(mu)),min(min(mu)),primal_ub-obj,sum(xi)),2,params.profile_log);

    tm = cputime;

    if params.profiling
        if tm > next_profile_tm
            next_profile_tm = next_profile_tm + params.profile_tm_interval;
            t_E = reshape(t_E,4,size(E,1)*m);
            [n_err,p_err,n_err_microlbl,p_err_microlbl] = compute_M3_error(WS,mu,Y_tr,loss,E,1,1);
            [microlabel_errors_ts,err_ts,time,Y_pred] = test_M3(mu,E,verbosity);
            tm = cputime;
            print_message(sprintf('tm: %d err_tr: %d (%3.2f) ml.loss tr: %d (%3.2f) err_ts: %d (%3.2f) ml loss ts: %d (%3.2f)',...
                round(tm-start_time),n_err,p_err*100,n_err_microlbl,p_err_microlbl*100,round(err_ts*size(Y_ts,1)),err_ts*100,sum(microlabel_errors_ts),sum(microlabel_errors_ts)/numel(Y_ts)*100),0,params.profile_log);
            print_message(sprintf('%d ',microlabel_errors_ts),0);
	    sfile = sprintf('Ypred_%d%d.mat',params.hloss,params.scaling);
                save(sfile,'Y_pred');
            Y_pred = []; t_E = reshape(t_E,4*size(E,1),m); 
        end
    end
    opt_round = opt_round + 1;



end


mu_E = mu(indices_XE);



function [mu_x,kxx_mu_x,obj,iter] = optimize_x(x,obj,mu_x,Kmu_x,kxx_mu_x,loss_x,te_x,C,maxiter)

global E;
global tolerance;
global Kx_tr;
global mu;
global m;
global Kmu;
global loss;

iter = 0;
while iter < maxiter

    gradient =  loss_x - Kmu_x;

    mu_1 = max_LP_direction_x(gradient,mu_x,x,C,0);

    d_x = mu_1 - mu_x;

    %if abs(sum(mu_1) - C*size(E,1)) < 1e-6
        smu_1_te = sum(reshape(mu_1.*te_x,4,size(E,1)));
        smu_1_te = reshape(smu_1_te(ones(4,1),:),length(mu_x),1);
        kxx_mu_1 = ~te_x+mu_1-smu_1_te;
        %else
        %kxx_mu_1 = 0;
        %end

    Kmu_1 = Kmu_x + kxx_mu_1 - kxx_mu_x;

    Kd_x = Kmu_1 - Kmu_x;
    l = gradient'*d_x; q = d_x'*Kd_x;
    alpha = min(l/q,1);

    delta_obj = gradient'*d_x*alpha - alpha^2/2*d_x'*Kd_x;


    if or(delta_obj < 0,alpha < 0)
        break;
    end

    mu_x = mu_x + d_x*alpha;
    Kmu_x = Kmu_x + Kd_x*alpha;
    obj = obj + delta_obj;
    kxx_mu_x = (1-alpha)*kxx_mu_x + alpha*kxx_mu_1;

    iter = iter + 1;
end

