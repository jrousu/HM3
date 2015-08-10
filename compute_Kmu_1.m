function compute_Kmu_1(Mu_1,pos_g,zero_g,neg_g,chunk)

global m;
global E;
global Kmu_1;
global Kx_tr;

global IndEdgeVal;


m_oup = size(Kx_tr,2);

if nargin < 3
    chunk = 1:m_oup;
end

term12 = zeros(size(E,1),m_oup);
Kmu_1 = zeros(4,size(E,1)*m);
fprintf('max chunk: %d min chunk: %d min pos: %d max pos: %d \n',max(chunk),min(chunk),min(pos_g),max(pos_g));
Kx_chunk = Kx_tr(pos_g,chunk);

% contribution of the examples with changed Mu_1 with non-zero components
for u = 1:4
  
    Mu_u = reshape(Mu_1(u,:),size(E,1),m);
    Mu_u = Mu_u(:,pos_g);
    H_u = full(IndEdgeVal{u}(:,pos_g)) - Mu_u;
    H_u = H_u*Kx_chunk;
    
    term12 = term12 + H_u.*IndEdgeVal{u}(:,chunk);
    Kmu_1(u,:) = reshape(-H_u,1,size(E,1)*m_oup);

end

term12 = reshape(term12,1,size(E,1)*m_oup);
for u = 1:4
    Kmu_1(u,:) = Kmu_1(u,:) + term12;
end
% contribution of the  examples Mu_1(x,:) = Mu_0(x,:)
if ~isempty(zero_g)
    Kmu_1 = Kmu_1 + compute_Kmu(Mu_1,Kx_tr,chunk,zero_g);
end









