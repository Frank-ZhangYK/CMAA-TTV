%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Yuanke Zhang, Southern Medical University & Qufu Normal University
% yuankezhang@163.com; lemonzyk@fmmu.edu.cn
% 2020.8


clear ;
close all;
path(path, 'tools');


%% setup target geometry
ig = image_geom('nx',256, 'ny', 256,'dx',256/256, 'offset_y',0, 'down',1);
sg = sino_geom('ge1', 'units', 'mm', 'strip_width', 'd', 'down', 1);
ig.mask = ig.circ > 0;
A = Gtomo2_dscmex(sg, ig,'nthread', jf('ncore')*2-1);

%% load data
printm('Loading external sinogram, weight, fbp...');
load ('./data/fbp_LD.fbp.mat');     % fbp with ramp filter
xfbp = fbp_LD;  

load('./data/sino_LD.sino.mat');     % sinogram data
sino = sino_LD; 

load('./data/invVar_LD.invVar.mat');
wi = invVar_LD;                      % inverse of the variance

printm('Pre-calculating denominator D_A...');
load('./data/denom_LD.denom.mat');   % diag{|A^T|W|A|1}
denomD = denom; 

%% reshape data
[imgM,imgN,p] = size(xfbp);
[sinoM,sinoN,p] = size(sino);
xfbpD = zeros(imgM*imgN,p);
sinoD = zeros(sinoM*sinoN,p);
wiD = zeros(sinoM*sinoN,p);
for ii = 1:p
    bandp = xfbp(:,:,ii);
    xfbpD(:,ii) = bandp(:);

    bandp = sino(:,:,ii);
    sinoD(:,ii) = bandp(:);

    bandp = wi(:,:,ii);
    wiD(:,ii) = bandp(:);
end

%% set parameter
nblock = 12;            % Subset Number, it can be set as a smaller value to save time
nIter = 2;              % Inner Iteration
nOuterIter =40;         % Outer Iteration
pixmax = inf;           % Set upper bond for pixel values

coe_rank=10;
r = [coe_rank,coe_rank,coe_rank]; % rank parameter
coe_tau = 2500;
tau_tmp = coe_tau *sqrt(ig.nx*ig.ny);
tau_rho =1.1; 
tau = [tau_tmp,tau_tmp,tau_rho*tau_tmp]; % tau parameter
stop_tol = 1e-3;
coe_mu = 1e5; % one could also initialize this Lagrangian parameter with a 
              % very small value and gradually increase it to avoid 
              % falling into local optimum. this would take longer time to converge.
mu_rho = 1;
max_mu = 1e5;
mu = [coe_mu,coe_mu,coe_mu];

%% pre-compute D_R
KK=4*ones(ig.nx,ig.ny); 
KK=KK(ig.mask);
KK=repmat(KK,p,1); 
D_R = sum(mu) * KK;

Ab = Gblock(A, nblock); clear A

%% Initializing optimization variables
sizeD = size(xfbp);
x_recD = xfbpD;

% Omega_x and Z_x initial
tv_x = diff_x(x_recD,sizeD);
tv_x = reshape(tv_x,[imgM*imgN,p]);
[Omega_x,S_x,Z_x] = svd(tv_x,'econ');
Omega_x = Omega_x(:,1:r(1))*S_x(1:r(1),1:r(1));
Z_x = Z_x(:,1:r(1));

% Omega_y and Z_y initial
tv_y = diff_y(x_recD,sizeD);
tv_y = reshape(tv_y,[imgM*imgN,p]);
[Omega_y,S_y,Z_y] = svd(tv_y,'econ');
Omega_y = Omega_y(:,1:r(2))*S_y(1:r(2),1:r(2));
Z_y = Z_y(:,1:r(2));

% Omega_z and Z_z initial
tv_z = diff_z(x_recD,sizeD);
tv_z = reshape(tv_z,[imgM*imgN,p]);
[Omega_z,S_z,Z_z] = svd(tv_z,'econ');
Omega_z = Omega_z(:,1:r(3))*S_z(1:r(3),1:r(3));
Z_z = Z_z(:,1:r(3));

% Gamma1, Gamma2 and Gamma3 initial
Gamma1 = zeros(size(x_recD));  % multiplier for Dx_X-U_x*V_x
Gamma2 = zeros(size(x_recD));  % multiplier for Dy_X-U_y*V_y
Gamma3 = zeros(size(x_recD));  % multiplier for Dz_X-U_z*V_z

info = struct('rank',r,'tau',tau,...
            'nIter',nIter,'nOuterIter',nOuterIter,'xrecD',[]);
        
xiniD = zeros(imgM,imgN,p);
xrecD_msk = zeros(size(Ab,2),p);
for ii = 1:p
    tmp = reshape(xfbpD(:,ii),[imgM,imgN]);
    xiniD(:,:,ii) = tmp.*ig.mask;
    xrecD_msk(:,ii) = tmp(ig.mask);
end
        
xrecD = xfbp;
SqrtPixNum = sqrt(p*sum(ig.mask(:)>0)); 
 
%% reconstruction
for i_CMAA_TTV = 1:nOuterIter  
    
    xoldD_msk = xrecD_msk;  
    
    %% updata x use fessler's os_rlalm algorithm 
    R = Reg_CMAA_TTV(ig.mask, sizeD,mu, Gamma1, Gamma2, Gamma3, Omega_x,Omega_y,Omega_z,Z_x,Z_y,Z_z);

    fprintf('Iteration = %d: \n', i_CMAA_TTV);
    xrecD_msk = pwls_os_rlalm_CMAA_TTV(xrecD_msk, Ab, sino, wi,  ...
                R, denomD, D_R, 'pixmax', pixmax, 'chat', 0, 'alpha', 1.999, 'rho', [], 'niter', nIter);
    for ii=1:p
        tmp = ig.embed(xrecD_msk(:,ii));
        xrecD(:,:,ii) = tmp;
    end
    info.xrecD = xrecD;
    figure(101); imshow([xfbp(:,:,17), xrecD(:,:,17)],[25 60]);
    
    %% -Update Omega_x, Omega_y and Omega_z 
    tmp_x = reshape(diff_x(xrecD,sizeD),[imgM*imgN,p]);
    tmp_x = tmp_x+Gamma1/mu(1);
    Omega_x = softthre(tmp_x*Z_x, tau(1)/mu(1));
    tmp_y = reshape(diff_y(xrecD,sizeD),[imgM*imgN,p]);
    tmp_y = tmp_y+Gamma2/mu(2);
    Omega_y = softthre(tmp_y*Z_y, tau(2)/mu(2));
    tmp_z = reshape(diff_z(xrecD,sizeD),[imgM*imgN,p]);
    tmp_z = tmp_z+Gamma3/mu(3);
    Omega_z = softthre(tmp_z*Z_z, tau(3)/mu(3)); 
    
    %% -Update Z_x, Z_y and Z_z
    [u,~,v] = svd(tmp_x'*Omega_x,'econ');
    Z_x = u*v';
    [u,~,v] = svd(tmp_y'*Omega_y,'econ');
    Z_y = u*v';
    [u,~,v] = svd(tmp_z'*Omega_z,'econ');
    Z_z = u*v'; 
    
    %% stop criterion  
    leq1 = reshape(diff_x(xrecD,sizeD),[imgM*imgN,p])- Omega_x*Z_x';
    leq2 = reshape(diff_y(xrecD,sizeD),[imgM*imgN,p])- Omega_y*Z_y';
    leq3 = reshape(diff_z(xrecD,sizeD),[imgM*imgN,p])- Omega_z*Z_z';
    normD = norm(reshape(xrecD,[imgM*imgN,p]), 'fro');
    abs_leq1 = abs(leq1);
    abs_leq2 = abs(leq2);
    abs_leq3 = abs(leq3);
    stopC1 = sum(abs_leq1(:))/normD;
    stopC2 = sum(abs_leq2(:))/normD;
    stopC3 = sum(abs_leq3(:))/normD;
    stopC4 = norm(xrecD_msk(:) - xoldD_msk(:)) / norm(xoldD_msk);
    disp(['iter ' num2str(i_CMAA_TTV)  ...
                ',relE=' num2str(stopC4,'%2.3e') ',|DX-Omega*Z|=' num2str(sum(abs_leq1(:)),'%2.3e')...
                ',|DY-Omega*Z|=' num2str(sum(abs_leq2(:)),'%2.3e')...
                ',|DZ-Omega*Z|=' num2str(sum(abs_leq3(:)),'%2.3e')]);
    if stopC1 < stop_tol &&  stopC2 < stop_tol && stopC3 < stop_tol && stopC4 < stop_tol
        break;
    else
        Gamma1 = Gamma1 + mu(1)*leq1;
        Gamma2 = Gamma2 + mu(2)*leq2;
        Gamma3 = Gamma3 + mu(3)*leq3;
        mu = min(max_mu,mu*mu_rho);             
    end
end

save(sprintf('./result/CMAA-TTV_tau%.1e_rank%g_MU%d.mat',  coe_tau, coe_rank, coe_mu), 'info')


