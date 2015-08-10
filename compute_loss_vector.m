function loss = compute_loss_vector(hierarchical,scaling,mlloss,testset)

global E;
global Y_tr;
global Y_ts;
global l;

if nargin < 4
    testset = 0;
end


loss_scaling = compute_loss_scaling(scaling,mlloss);

if testset
    Te1 = Y_ts(:,E(:,1))'; % the label of edge parent
    Te2 = Y_ts(:,E(:,2))'; % the label of edge child    
    m = size(Y_ts,1);
    Loss = zeros(m*size(E,1),4);
else
    Te1 = Y_tr(:,E(:,1))'; % the label of edge parent
    Te2 = Y_tr(:,E(:,2))'; % the label of edge child 
    m = size(Y_tr,1);
    Loss = zeros(m*size(E,1),4);
end


u = 0;
for u_1 = 1:2
    
    for u_2 = 1:2
        
        u = u + 1;
    
        if hierarchical % only penalize child when parent was correct
            if mlloss  % take part ofthe blame of parents mistake
              Loss(:,u) = reshape(diag(loss_scaling(E(:,1)))*(Te1 ~= u_1),m*size(E,1),1);
            end
            Loss(:,u) = Loss(:,u) + reshape(diag(loss_scaling(E(:,2)))*and(Te1 == u_1,Te2 ~= u_2),m*size(E,1),1);
        else % always pay both for parents and child's mistakes
            %Loss(:,u) = reshape(sparse(diag(loss_scaling(E(:,1))))*(Te1 ~= u_1)+sparse(diag(loss_scaling(E(:,2))))*(Te2 ~= u_2),m*size(E,1),1);            
            Loss(:,u) = reshape(diag(loss_scaling(E(:,1)))*(Te1 ~= u_1)+diag(loss_scaling(E(:,2)))*(Te2 ~= u_2),m*size(E,1),1);
        end
    end
end

loss = reshape(Loss',4*size(E,1)*m,1);