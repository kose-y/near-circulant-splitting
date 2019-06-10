function x = CT_ADMM_3D(N,M,  iters, lambda, alpha, beta, sino,  A, mask)

sino = reshape(sino, [], 1);
rng(999);
x = randn(N, N, M);
u = zeros(size(A, 1), 1);
vx = zeros(N, N, M);
vy = zeros(N, N, M);
vz = zeros(N, N, M);

etau = zeros(size(u));
etavx = zeros(size(vx));
etavy = zeros(size(vy));
etavz = zeros(size(vz));

Dxx = zeros(N, N, M);
Dxy = zeros(N, N, M);
Dxz = zeros(N, N, M);

iter_vec = [];
err_vec_ADMM = [];
inner_iters = 0;

H = ones(N, N, M);

tic
for ii=1:(iters/10)
    
    disp(ii)
    xprime = x;

    Dpv = compute_Dpv(vx - etavx, vy - etavy, vz - etavz, N, M);

    Gx_tgt = embed(A' * double(u - etau), mask) + beta * Dpv;

    % solve for x.
    [x, kk] = cgsolve(x, Gx_tgt, N, M, A, mask, beta, H);

    inner_iters = inner_iters + kk; 

    Ax = A * x(mask(:));

    % step 4
    u = 1.0/(1.0 + alpha) * (sino + alpha * (Ax + etau));

    % step 5
    [Dxx, Dxy, Dxz] = compute_Dx(x, N, M);

    rhox = Dxx + etavx; 
    rhoy = Dxy + etavy;
    rhoz = Dxz + etavz;

    vx = sign(rhox) .* max(abs(rhox) - lambda/(alpha*beta), 0);
    vy = sign(rhoy) .* max(abs(rhoy) - lambda/(alpha*beta), 0);
    vz = sign(rhoz) .* max(abs(rhoz) - lambda/(alpha*beta), 0);

    % steps 6-8
    etau  = etau -  (u  - Ax);
    etavx = etavx - (vx - Dxx); 
    etavy = etavy - (vy - Dxy);
    etavz = etavz - (vz - Dxz);

    
    if mod(inner_iters, 10) == 0
        disp((1/2)*gather(sum(sum(sum((A*x(mask(:))-sino).^2)))))
        disp(lambda*gather(sum(sum(sum(abs(x(1:N,1:(N-1),1:M)-x(1:N,2:N,1:M)))))+sum(sum(sum(abs(x(1:(N-1),1:N,1:M)-x(2:N,1:N,1:M)))))+sum(sum(sum(abs(x(1:N,1:N,1:(M-1))-x(1:N,1:N,2:M)))))));
    end


    if mod(inner_iters, 10) == 0
        disp(inner_iters)
        filename_x = sprintf('admm_%03d_x.png', ii);
        filename_y = sprintf('admm_%03d_y.png', ii);
        filename_z = sprintf('admm_%03d_z.png', ii);
        imwrite(squeeze(x(210,:,:))'/1.7e+03, filename_x);
        imwrite(squeeze(x(:,210,:))'/1.7e+03, filename_y);
        imwrite(x(:,:,66)/1.7e+03, filename_z);
    end
    
end
toc

function [x, kk] = cgsolve(xin, b, N, M, A, mask, beta, precond)
  x = xin;
  r = b - compute_Gx(x, N, M, A, mask, beta);
  rsold = sum(sum(sum(r.^2)));
  
  p = real(ifftn(precond .* fftn(r)));
  z = p;
  rtz = sum(sum(sum(r .* z)));
  for kk = 1:10
    Gp = compute_Gx(p, N, M, A, mask, beta);
    alpha = rtz / sum(sum(sum(p .* Gp)));
    x = x + alpha * p;
    r = r - alpha * Gp;
    rsnew = sum(sum(sum(r .^ 2)));

    z = real(ifftn(precond .* fftn(r)));

    rtzold = rtz;
    rtz = sum(sum(sum(r .* z)));
    beta = rtz / rtzold;
    p = z + beta * p;
    
    %disp(sum(sum(sum((compute_Gx(x, N, M, A, mask, beta) - b).^2))))
  end
end

function Gx = compute_Gx(x, N, M, A, mask, beta)
  [Dxx, Dxy, Dxz] = compute_Dx(x, N, M);
  Gx = embed(A' * (A * x(mask(:))), mask) + beta * compute_Dpv(Dxx, Dxy, Dxz, N, M);
end

function [Dxx, Dxy, Dxz] = compute_Dx(x, N, M)
  Dxx = zeros(N, N, M);
  Dxy = zeros(N, N, M);
  Dxz = zeros(N, N, M);
  Dxx(1:N, 1:(N-1), 1:M) = x(1:N, 1:(N-1), 1:M) - x(1:N, 2:N, 1:M);
  Dxy(1:(N-1), 1:N, 1:M) = x(1:(N-1), 1:N, 1:M) - x(2:N, 1:N, 1:M);
  Dxz(1:N, 1:N, 1:(M-1)) = x(1:N, 1:N, 1:(M-1)) - x(1:N, 1:N, 2:M);
end

function Dpv = compute_Dpv(vx, vy, vz, N, M)
  Dpv = vx + vy + vz;
    Dpv(1:N,2:N,1:M) = Dpv(1:N,2:N,1:M) - vx(1:N,1:(N-1),1:M);
    Dpv(2:N,1:N,1:M) = Dpv(2:N,1:N,1:M) - vy(1:(N-1),1:N,1:M);
    Dpv(1:N,1:N,2:M) = Dpv(1:N,1:N,2:M) - vz(1:N,1:N,1:(M-1)); 
end

end
