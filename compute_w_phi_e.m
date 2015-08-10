function w_phi_e = compute_w_phi_e(Kx)

global E;
global m;
global IndEdgeVal;
global t_E;
global mu;

m_oup = size(Kx,2);

w_phi_e = sum(mu);
w_phi_e = w_phi_e(ones(4,1),:);
w_phi_e = t_E.*w_phi_e;
w_phi_e = w_phi_e-mu;
w_phi_e = reshape(w_phi_e,4*size(E,1),m);
w_phi_e = w_phi_e*Kx;

w_phi_e = reshape(w_phi_e,4,size(E,1)*m_oup);

