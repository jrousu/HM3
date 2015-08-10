function [Ind,Ind2,Ind3] = precomputeIndicatorMatrices(edomsize)

global Ind;
global Ind2;
global Ind3;

Ind = cell(edomsize,edomsize); % a cell matric for each pair (t_i,t_j)x(y_i,y_j)
Ind2 = cell(edomsize^2,1);

onesd = ones(edomsize,1);

% K_x_xhat is the "kernel matrix" for Delta kij(x,y_i,y_j,x^hat,
% y_i^hat,y_j^hat)
I = eye(edomsize);
Ones = ones(edomsize);


for h = 1:edomsize
    
    for hhat = 1:edomsize

        Ind{h,hhat} = I;
        
        
        Ind{h,hhat}(hhat,:) = Ind{h,hhat}(hhat,:) - onesd';
        Ind{h,hhat}(:,h) = Ind{h,hhat}(:,h) - onesd;

        if h == hhat
            Ind{h,hhat} = Ind{h,hhat} + Ones;
        end
        
        h2 = (h-1)*edomsize+hhat;
        Ind2{h2} = Ind{h,hhat};
        Ind3(:,:,h2) = Ind2{h2};
    end
end  

