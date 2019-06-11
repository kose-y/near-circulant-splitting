close all; clear all;
N = 512;
xtrue = imread('data/mayo_clinic_image.jpg');
xtrue = double(xtrue(:, :, 1));
%xtrue = phantom('Modified Shepp-Logan',N);

lambda = 1;
iters = 100000;
useGPU = true;

if useGPU
    toGPU = @(x) gpuArray(x);
else
    toGPU = @(x) x;
end

im = toGPU(xtrue);
th = linspace(0,180,61);
th = th(1:(length(th)-1));
sino = toGPU(radon(im, th));


%NCS experiment
alpha = 0.01;
beta = 0.01;
gamma = 1;

rng(999)
x = toGPU(randn(N,N));
u = 0*toGPU(radon(x,th));
vx = toGPU(zeros(N,N));
vy = toGPU(zeros(N,N));


[kk,ll] = meshgrid(0:(N-1),0:(N-1));
kk = toGPU(kk); ll = toGPU(ll);
H =  gamma*ones(N,N) + 300*alpha./(sqrt(min(kk,N-kk).^2+min(ll,N-ll).^2))  +4* beta^2/alpha*((sin(kk*pi/N)).^2+(sin(ll*pi/N)).^2);
H(1,1)=10;
H = 1./H;
H = toGPU(H);
    
err_vec_NCS = zeros(iters,1);

tic
for ii=1:iters
    %disp(ii)
    xprime = x; %save previous iterate
    
    Dpv = vx+vy;
    Dpv(1:N,2:N) = Dpv(1:N,2:N) - vx(1:N,1:(N-1));
    Dpv(2:N,1:N) = Dpv(2:N,1:N) - vy(1:(N-1),1:N);
    
    y = iradon(u,th,'none',N)+beta/alpha*Dpv;
    x = x-real(ifft2(H.*fft2(y)));
    
    xprime = (2*x-xprime);
    r = radon(xprime, th);
    u = 1/(1+alpha)*(u+alpha*(r-sino));
    lapb = lambda*alpha/beta;
    vx(1:N,1:(N-1)) = max(min(vx(1:N,1:(N-1)) + beta * (xprime(1:N,1:(N-1))-xprime(1:N,2:N)),lapb),-lapb);
    vy(1:(N-1),1:N) = max(min(vy(1:(N-1),1:N) + beta * (xprime(1:(N-1),1:N)-xprime(2:N,1:N)),lapb),-lapb);

    err_vec_NCS(ii)=(1/2)*gather(sum(sum((radon(x, th)-sino).^2)))+lambda*gather(sum(sum(abs(x(1:N,1:(N-1))-x(1:N,2:N))))+sum(sum(abs(x(1:(N-1),1:N)-x(2:N,1:N)))));
    %disp(err_vec_NCS(ii))
end
toc
x_ncs = x;
clear x u vx vy


%PDHG experiment
alpha = 0.01;
beta = 0.03;
gamma = 10;

rng(999)
x = toGPU(randn(N,N));
u = 0*toGPU(radon(x,th));
vx = toGPU(zeros(N,N));
vy = toGPU(zeros(N,N));

err_vec_PDHG = zeros(iters,1);

tic
for ii=1:iters
    %disp(ii)
    xprime = x; %save previous iterate
    
    Dpv = vx+vy;
    Dpv(1:N,2:N) = Dpv(1:N,2:N) - vx(1:N,1:(N-1));
    Dpv(2:N,1:N) = Dpv(2:N,1:N) - vy(1:(N-1),1:N);
    
    y = iradon(u,th,'none',N)+beta/alpha*Dpv;
    x = x-1/gamma*y;
    
    xprime = (2*x-xprime);
    r = radon(xprime, th);
    u = 1/(1+alpha)*(u+alpha*(r-sino));
    lapb = lambda * alpha/beta;
    vx(1:N,1:(N-1)) = max(min(vx(1:N,1:(N-1)) + beta * (xprime(1:N,1:(N-1))-xprime(1:N,2:N)),lapb),-lapb);
    vy(1:(N-1),1:N) = max(min(vy(1:(N-1),1:N) + beta * (xprime(1:(N-1),1:N)-xprime(2:N,1:N)),lapb),-lapb);

    err_vec_PDHG(ii)=(1/2)*gather(sum(sum((radon(x, th)-sino).^2)))+lambda*gather(sum(sum(abs(x(1:N,1:(N-1))-x(1:N,2:N))))+sum(sum(abs(x(1:(N-1),1:N)-x(2:N,1:N)))));
    %disp(err_vec_PDHG(ii))
end
toc
x_pdhg = x;
clear x u vx vy


%ADMM experiment
alpha = 1;
beta = 0.003;

xtmp = toGPU(zeros(N, N));
center = floor(N/2);
xtmp(center, center) = 1;
precond = compute_Gx(xtmp, N, th, beta, useGPU);
precond = fft2(precond);
%beta = max(max(abs(precond)))/min(min(abs(precond)))/100

rng(999)
x = toGPU(randn(N,N));
u = 0*toGPU(radon(x,th));
%vx = toGPU(zeros(N,N));
%vy = toGPU(zeros(N,N));
[vx, vy] = compute_Dx(x, N, useGPU);
etau = toGPU(zeros(size(u)));
etavx = toGPU(zeros(size(vx)));
etavy = toGPU(zeros(size(vy)));
Dxx = toGPU(zeros(N,N));
Dxy = toGPU(zeros(N,N));
H = toGPU(ones(size(x)));
tic
iter_vec = [];
err_vec_ADMM = [];
inner_iters = 0;
for ii=1:(iters/10)
    %disp(ii)
    xprime = x; %save previous iterate
    
    % step 3
    
    Dpv = compute_Dpv(vx-etavx, vy-etavy, N);
    
    Gx = iradon(u-etau, th, 'none', N) + beta * Dpv;
    
    %solve for x
    [x, kk] = cgsolve(x, Gx, N, th, beta, useGPU, H);
    
    inner_iters = inner_iters + kk;
    
    Ax = radon(x, th);
    % step 4
    u = 1.0/(1.0+alpha) * (sino + alpha * (Ax + etau));
    
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
    err_vec_ADMM = [err_vec_ADMM, (1/2)*gather(sum(sum((radon(x, th)-sino).^2)))+lambda*gather(sum(sum(abs(x(1:N,1:(N-1))-x(1:N,2:N))))+sum(sum(abs(x(1:(N-1),1:N)-x(2:N,1:N)))))];
    %disp(err_vec_ADMM(end))
end
toc
x_admm = x;

save('par_beam_results.mat','err_vec_PDHG','err_vec_NCS','err_vec_ADMM','iter_vec')


%%
close all; clear all;
load('par_beam_results.mat')
minval = min([min(err_vec_NCS),min(err_vec_PDHG),min(err_vec_ADMM)]);
loglog(1:length(err_vec_NCS),err_vec_NCS-minval,'k','LineWidth',2)

hold on;
loglog(1:length(err_vec_PDHG),err_vec_PDHG-minval,'r--','LineWidth',2)
loglog(iter_vec,err_vec_ADMM-minval,'b:','LineWidth',2)

legend('NCS','PDHG','ADMM')
xlabel('Iterations')
ylabel('Objective value suboptimality')

pbaspect([2 1 1])
%ylim([1e3,3e8])

ax = gca;
ax.OuterPosition(3)=ax.OuterPosition(4);
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height*1.1];

set(gcf, 'Position', [100, 100, 500, 290])

title('Parallel beam experiments')
saveas(gcf,'par_plot.png')

%%
x_ncs_scaled = gather((x_ncs - min(min(xtrue)))/max(max(xtrue)));
x_pdhg_scaled = gather((x_pdhg - min(min(xtrue)))/max(max(xtrue)));
x_admm_scaled = gather((x_admm - min(min(xtrue)))/max(max(xtrue)));

figure
subplot(1,3,1)
imshow(x_ncs_scaled)
title('Parallel beam (NCS)')
subplot(1,3,2)
imshow(x_pdhg_scaled)
title('Parallel beam (PDHG)')
subplot(1,3,3)
imshow(x_admm_scaled)
title('Parallel beam (ADMM)')

set(gcf, 'Position', [100, 100, 800, 300])

imwrite(x_ncs_scaled, 'par_beam_ncs.png');
imwrite(x_pdhg_scaled, 'par_beam_pdhg.png');
imwrite(x_admm_scaled, 'par_beam_admm.png');


%%
function [x, kk] = cgsolve(xin, b, N, th, beta, useGPU, precond)
  x = xin;
  r = b - compute_Gx(x, N, th, beta, useGPU);
  p = real(ifft2(fft2(r).*precond));
  z = p;
  rtz = sum(sum(r.*z));
  for kk = 1:10
    Gp = compute_Gx(p, N, th, beta, useGPU);
    alpha = rtz / sum(sum(p .* Gp));
    
    x = x + alpha * p;
    r = r - alpha * Gp; 
    rsnew = sum(sum(r .^ 2));
    if sqrt(rsnew) < 1e-5
      break;
    end
    z = real(ifft2(fft2(r).*precond));
    rtzold = rtz;
    rtz = sum(sum(r.*z));
    beta = rtz/rtzold;
    p = z + beta * p;
    %disp(sum(sum((compute_Gx(x, N, th, beta, useGPU) - b).^2)))
  end 
end

function Gx = compute_Gx(x, N, th, beta, useGPU)
  [Dxx, Dxy] = compute_Dx(x, N, useGPU);
  Gx = iradon(radon(x, th), th, 'none', N) + beta * compute_Dpv(Dxx, Dxy, N);
end

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

function dpv = compute_Dpv(vx, vy, N)
  dpv = vx + vy;
  dpv(1:N, 2:N) = dpv(1:N, 2:N) - vx(1:N, 1:(N-1));
  dpv(2:N, 1:N) = dpv(2:N, 1:N) - vy(1:(N-1), 1:N);
end
