%This code only works on MacOS and Linux, not on Windows.
%(Many features of MIRT toolbox do not work on Windows.)

close all; clear all;
addpath('../ADMM_nonnegativity/')
% read XCAT data 
%bin_read is a function provided by ADMM_nonnegativity
mumap = bin_read('../ADMM_nonnegativity/Y90PET_atn_1.bin'); % reading attenuation map 
mumap = mumap / 10; % cm -> mm
ind = bin_read('../ADMM_nonnegativity/Y90PET_act_1.bin');
liver = single((ind == 13)); % setting liver mask 
llung = (ind == 15); rlung = (ind == 16); % setting lung mask 
lung = single(llung + rlung);


% image/projection geometry setting
nx = 128; % number of voxel in x direction 
ny = 128; % number of voxel in y direction 
nz = 100; % number of voxel in z direction 
dx = 4; % voxel size in x direction
dz = 4; % voxel size in z direction
na = 168; % number of projection angle

% recon parameter setting
rho = 1; % initial rho value 
patient_A = 1; % 1 if Patient A condition, 0 if Patient B condition
beta = 2^-3; % regularization parameter 
alpha = 1; % constraint: Ax + alpha * r > 0 
psi = 1; % negml poisson-gaussian switching parameter 

xiter = 1; % number of x inner update 
niter = 50; % number of admm update 
titer = xiter * niter; % total number of iterations 

% setting 
ig = image_geom('nx', nx, 'ny', ny, 'nz', nz, 'dx', dx, 'dz', dz);
ig.mask = ig.circ(ig.dx * (ig.nx/2-2), ig.dy * (ig.ny/2-4)) > 0;
sg = sino_geom('par', 'nb', ig.nx, 'na', na * ig.nx / nx, ...
    'dr', ig.dx, 'strip_width', 2*ig.dx);
R = Reg1(ig.mask, 'edge_type', 'tight', ...
    'beta', beta, 'pot_arg', {'quad'},'type_denom', 'matlab');

% Patient A or B condition 
if patient_A
    f.counts = 6e5;
    f.scatter_percent = 500;
else % We use smaller area of liver in the Patient B case
    f.counts = 1e5;
    f.scatter_percent = 1800;
    liver(64:128,:,:) = 0;
    liver(:,76:128,:) = 0;
    liver(:,1:50,:) = 0;
    liver(:,:,1:55) = 0;
    liver(:,:,73:end) = 0;
end

f.scatter_percent = 0;

% setting hot spot/cold spot/healthy liver/eroded liver
hotspot = ellipsoid_im(ig, ...
    [-40 30 50  20 20 25    0 0 1;
    ], 'oversample', 1);
coldspot = ellipsoid_im(ig, ...
    [-90 -10 60  20 20 25    0 0 1;
    ], 'oversample', 1);
hliver = liver - hotspot - coldspot;
hliver(hliver < 0) = 0;
se = strel('arbitrary',eye(2));
eliver = imerode(hliver,se);

% true image 
xtrue = double(liver + 4*hotspot - coldspot + 0.04*lung);
xtrue(xtrue < 0) = 0;

% system model
f.dir = test_dir;
f.dsc = [test_dir 't.dsc'];
f.wtr = strrep(f.dsc, 'dsc', 'wtr');
f.mask = [test_dir 'mask.fld'];
fld_write(f.mask, ig.mask)

tmp = Gtomo2_wtmex(sg, ig, 'mask', ig.mask_or);

[tmp dum dum dum dum is_transpose] = ...
    wtfmex('asp:mat', tmp.arg.buff, int32(0));
if is_transpose
    tmp = tmp'; % because row grouped
end
delete(f.wtr)
wtf_write(f.wtr, tmp, ig.nx, ig.ny, sg.nb, sg.na, 'row_grouped', 1)

f.sys_type = sprintf('2z@%s@-', f.wtr);



load('PET128.mat') % we use matrix E for 128x128 design
d = 128;  %Number of detectors
N = 128;  %Image is NxN
p = N^2;  %Number of pixels
q = d*(d-1)/2;  %number of detector pairs

% created noisy count data based on our E matrix
xtrue_2d = squeeze(xtrue(:, :, 66));
mumap_2d = squeeze(mumap(:, :, 66));
ytrue = E * reshape(double(xtrue_2d), [p, 1]);
li = E * reshape(double(mumap_2d), [p,1]);
ci = exp(-li);
ci = ci * f.counts / sum(col(ci.*ytrue));
ytrue = ci .* ytrue; 
ri = ones(size(ytrue)) * f.scatter_percent / 100 * mean(ytrue(:));
ci = ones(size(ri));
y = poisson(ytrue + ri);



iters = 10000;
useGPU = true;
lambda = 0.001;

if useGPU
    toGPU = @(x) gpuArray(x);
else
    toGPU = @(x) x;
end
E = toGPU(E);
Et = transpose(E);%toGPU(Et);
y = toGPU(double(y));



%NCS experiment
alpha = 0.002;
beta = 0.002;
gamma = 0.001;



vx = toGPU(zeros(N,N));
vy = toGPU(zeros(N,N));

rng(999)
x = toGPU(100*rand(N,N));
u = toGPU(zeros(q,1));


[kk,ll] = meshgrid(0:(N-1),0:(N-1));
kk = toGPU(kk);
ll = toGPU(ll);
H =  gamma*ones(N,N) + alpha * 64./(sqrt(min(kk,N-kk).^2+min(ll,N-ll).^2))  + beta^2/alpha*(4*(sin(kk*pi/N)).^2+4*(sin(ll*pi/N)).^2);
H(1,1)=1/(10);
H = 1./H;
%how to handle the DC component is important

err_vec_NCS = zeros(iters,1);

tic
for ii=1:iters
    disp(ii)
    xprime = x;
    
    
    Dpv = vx+vy;
    Dpv(1:N,2:N) = Dpv(1:N,2:N) - vx(1:N,1:(N-1));
    Dpv(2:N,1:N) = Dpv(2:N,1:N) - vy(1:(N-1),1:N);
    
    yy =reshape(Et*u,[N,N])+beta/alpha*Dpv;
    
    x = x-real(ifft2(H.*fft2(yy)));
    
    
    xprime = 2*x-xprime;
    
    uprime = u + alpha*E*reshape(xprime, [p,1]);
    c = alpha*y;
    u = 1 + ((uprime-1) - sqrt((uprime-1).^2 +4*c))/2;
    %S = @(u,c) 1 + ((u-1)-sqrt((u-1).^2+4*c))/2;
    %u = S(u+alpha*E*reshape(xprime,[p,1]), alpha*y);
    
    lapb = lambda*alpha/beta;
    vx(1:N,1:(N-1)) = max(min(vx(1:N,1:(N-1)) + beta * (xprime(1:N,1:(N-1))-xprime(1:N,2:N)),lapb),-lapb);
    vy(1:(N-1),1:N) = max(min(vy(1:(N-1),1:N) + beta * (xprime(1:(N-1),1:N)-xprime(2:N,1:N)),lapb),-lapb);
   
    
    vv=max(E*reshape(x, [p,1]),1e-50);
    err_vec_NCS(ii)=gather(sum(vv-y.*log(vv)))+gather(lambda*(sum(sum(abs(x(1:N,1:(N-1))-x(1:N,2:N))))+sum(sum(abs(x(1:(N-1),1:N)-x(2:N,1:N))))));
end
toc
x_ncs = gather(x);



%PDHG experiment
alpha = 0.01;
beta = 0.01;
gamma = 0.03;

vx = toGPU(zeros(N,N));
vy = toGPU(zeros(N,N));

rng(999)
x = toGPU(100*rand(N,N));
u = toGPU(zeros(q,1));


H =  gamma*ones(N,N);
H = 1./H;


err_vec_PDHG = zeros(iters,1);

tic
for ii=1:iters
    disp(ii)
    xprime = x;
    
    
    Dpv = vx+vy;
    Dpv(1:N,2:N) = Dpv(1:N,2:N) - vx(1:N,1:(N-1));
    Dpv(2:N,1:N) = Dpv(2:N,1:N) - vy(1:(N-1),1:N);
    
    yy =reshape(Et*u,[N,N])+beta/alpha*Dpv;
    
    x = x-1/gamma*yy;
    
    
    xprime = 2*x-xprime;
    
    uprime = u + alpha*E*reshape(xprime, [p,1]);
    c = alpha*y;
    u = 1 + ((uprime-1) - sqrt((uprime-1).^2 +4*c))/2;
    %S = @(u,c) 1 + ((u-1)-sqrt((u-1).^2+4*c))/2;
    %u = S(u+alpha*E*reshape(xprime,[p,1]), alpha*y);
    
    lapb = lambda*alpha/beta;
    vx(1:N,1:(N-1)) = max(min(vx(1:N,1:(N-1)) + beta * (xprime(1:N,1:(N-1))-xprime(1:N,2:N)),lapb),-lapb);
    vy(1:(N-1),1:N) = max(min(vy(1:(N-1),1:N) + beta * (xprime(1:(N-1),1:N)-xprime(2:N,1:N)),lapb),-lapb);
    
    vv=max(E*reshape(x, [p,1]),1e-50);
    err_vec_PDHG(ii)=gather(sum(vv-y.*log(vv)))+gather(lambda*(sum(sum(abs(x(1:N,1:(N-1))-x(1:N,2:N))))+sum(sum(abs(x(1:(N-1),1:N)-x(2:N,1:N))))));
    disp(err_vec_PDHG(ii))
end
x_pdhg = gather(x);




alpha = 0.0003;
beta = 0.3;

xtmp = toGPU(zeros(N, N));
center = floor(N/2);
xtmp(center, center) = 1;
precond = compute_Gx(xtmp, N, E, Et, beta, useGPU);
precond = fft2(precond);

rng(999)
x = toGPU(100*rand(N));
u = toGPU(E * reshape(x, [p,1]));
%vx = toGPU(zeros(N,N));
%vy = toGPU(zeros(N,N));
[vx, vy] = compute_Dx(x, N, useGPU);
etau = toGPU(zeros(size(u)));
etavx = toGPU(zeros(size(vx)));
etavy = toGPU(zeros(size(vy)));


iter_vec = [];
err_vec_ADMM = [];
inner_iters = 0;
tic
for ii=1:(iters/10)
    disp(ii)
    %save previous iterate
    xprime = x;
    
    Dpv = compute_Dpv(vx-etavx, vy-etavy, N);
    
    Gx = reshape(Et * u, [N, N]) + beta * Dpv;
    % now solve for x.
    
    [x, kk] = cgsolve(x, Gx, N, E, Et, beta, useGPU, H);
    
    inner_iters = inner_iters + kk;
    
    Ax = E * reshape(x, [p,1]);
    % step 4
    u = 1.0/(1.0+alpha) * (y + alpha * (Ax + etau));
    
    % step 5
    [Dxx, Dxy] = compute_Dx(x, N, useGPU);
    %Dxx(1:N, 1:(N-1)) = x(1:N, 1:(N-1)) - x(1:N, 2:N);
    %Dxy(1:(N-1), 1:N) = x(1:(N-1), 1:N) - x(2:N, 1:N);
    rhox = Dxx + etavx;
    rhoy = Dxy + etavy;
    
    
    vx = sign(rhox) .* max(abs(rhox) - lambda/(alpha*beta), 0);
    vy = sign(rhoy) .* max(abs(rhoy) - lambda/(alpha*beta), 0);
    
    % steps 6-8
    etau  = etau  - (u  - Ax);
    etavx = etavx - (vx - Dxx);
    etavy = etavy - (vy - Dxy);
    
    iter_vec = [iter_vec inner_iters];
    vv=max(E*reshape(x, [p,1]),1e-50);
    err_vec_ADMM = [err_vec_ADMM, gather(sum(vv-y.*log(vv)))+gather(lambda*(sum(sum(abs(x(1:N,1:(N-1))-x(1:N,2:N))))+sum(sum(abs(x(1:(N-1),1:N)-x(2:N,1:N))))))];
    disp(err_vec_ADMM(end))
end
toc
x_admm = gather(x);



x_ncs = x_ncs - min(min(x_ncs));
x_pdhg = x_pdhg - min(min(x_pdhg));
x_admm = x_admm - min(min(x_admm));
maxval = max([max(max(x_ncs)),max(max(x_pdhg)),max(max(x_admm))]);

imwrite(x_ncs/maxval, 'PET_NCS.png');
imwrite(x_pdhg/maxval, 'PET_PDHG.png');
imwrite(x_pdhg/maxval, 'PET_ADMM.png');



function [Dxx, Dxy] = compute_Dx(x, N, useGPU)
  Dxx = zeros(N, N);
  Dxy = zeros(N, N);
  if useGPU
    Dxx = gpuArray(Dxx);
    Dxy = gpuArray(Dxy);
  end
  Dxx(1:N, 1:(N-1)) = x(1:N, 1:(N-1)) - x(1:N, 2:N);
  Dxy(1:(N-1), 1:N) = x(1:(N-1), 1:N) - x(2:N, 1:N);
end
function Gx = compute_Gx(x, N, E, Et, beta, useGPU)
    [Dxx, Dxy] = compute_Dx(x, N, useGPU);
    Gx = reshape(Et * (E * reshape(x, [N^2, 1])),[N, N]) + beta * compute_Dpv(Dxx, Dxy, N);
end
  function dpv = compute_Dpv(vx, vy, N)
    dpv = vx + vy;
    dpv(1:N, 2:N) = dpv(1:N, 2:N) - vx(1:N, 1:(N-1));
    dpv(2:N, 1:N) = dpv(2:N, 1:N) - vy(1:(N-1), 1:N);
  end

function [x, kk] = cgsolve(xin, b, N, E, Et, beta, useGPU, precond)
  x = xin+0.0001;
  r = b - compute_Gx(x, N, E, Et, beta, useGPU);
  rsold = sum(sum(r.^2));
  precond = precond;
  %p = r;
  p = real(ifft2(fft2(r).*precond));
  z = p;
  rtz = sum(sum(r.*z));
  for kk = 1:10
    Gp = compute_Gx(p, N, E, Et, beta, useGPU);
    alpha = rtz / sum(sum(p .* Gp));
    
    x = x + alpha * p;
    r = r - alpha * Gp; 
    rsnew = sum(sum(r .^ 2));
    if sqrt(rsnew) < 1e-5
      break;
    end
    %z = r;
    z = real(ifft2(fft2(r).*precond));
    rtzold = rtz;
    rtz = sum(sum(r.*z));
    beta = rtz/rtzold;
    p = z + beta * p;
    %disp(sum(sum((compute_Gx(x, N, th, beta, useGPU) - b).^2)))

  end 
end



