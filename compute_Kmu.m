function Kmu = compute_Kmu(Mu,Kx,chunk,ws)

global E;
global IndEdgeVal;

m_oup = size(Kx,2);
m = size(Kx,1);

if nargin < 3
    chunk = 1:m_oup;
end
if nargin < 4
    ws = 1:m;
end

Smu = reshape(sum(Mu),size(E,1),m);
Smu = Smu(:,ws);
term12 =zeros(1,size(E,1)*length(chunk));
Kmu = zeros(4,size(E,1)*length(chunk));

for u = 1:4
    IndEVu = full(IndEdgeVal{u});
    
    Rmu_u = reshape(Mu(u,:),size(E,1),m);
    Rmu_u = Rmu_u(:,ws);

    H_u = Smu.*IndEVu(:,ws);
    H_u = H_u - Rmu_u;
    Q_u = H_u*Kx(ws,chunk);
    
    term12 = term12 + reshape(Q_u.*IndEVu(:,chunk),1,length(chunk)*size(E,1));
    Kmu(u,:) = reshape(-Q_u,1,length(chunk)*size(E,1));
end

for u = 1:4
    Kmu(u,:) = Kmu(u,:) + term12;
end









