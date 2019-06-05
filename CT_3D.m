function x = CT_3D(N,M,  iters, lambda, alpha, beta, gamma, sino,  A, mask, ncs)
  rng(50000)
  useGPU=false;
  if useGPU
    toGPU = @(x) gpuArray(x);
  else 
    toGPU = @(x) x;
  end
  %im = toGPU(phantom('Modified Shepp-Logan',N));
  
  %im = toGPU(im);
  mask = toGPU(mask);
  %imv = im(:);

  sino = reshape(sino, [], 1);


 % lambda = 1;
  x = toGPU(zeros(N,N, M));
  %u = 0*toGPU(radon(x,th));
  u = toGPU(zeros(size(A, 1), 1));
  vx = toGPU(zeros(N,N,M));
  vy = toGPU(zeros(N,N,M));
  vz = toGPU(zeros(N,N,M));
  A = A;


  if ~ncs
    % Chambolle-Pock filter
   
    %alpha = .002;
    %beta = 500;
    H =  gamma*toGPU(ones(N,N,M));
    H = 1./H;
  else

  % Unnamed filter

%  alpha = 0.1;
%  beta = 100;
    [kk,ll, mm] = meshgrid(0:(N-1),0:(N-1), 0:(M-1));
    kk = toGPU(kk); ll = toGPU(ll);
    %H =  gamma*ones(N,N,M) + alpha*183/2./(sqrt(min(kk,N-kk).^2+min(ll,N-ll).^2 + min(mm, M-mm).^2*(N^2)/(M^2)))  + beta^2/alpha*(4*(sin(kk*pi/N)).^2+4*(sin(ll*pi/N)).^2 + 4 * (sin(mm*pi/M)).^2);
    H =  gamma*ones(N,N,M) + alpha*182/2./(sqrt(min(kk,N-kk).^2+min(ll,N-ll).^2 + min(mm, M-mm).^2*(N^2)/(M^2)))  + beta^2/alpha*(4*(sin(kk*pi/N)).^2+4*(sin(ll*pi/N)).^2 + 4 * (sin(mm*pi/M)).^2);
    H(1,1,1)=400;
    H = toGPU(1./H);
    %how to handle the DC component is important
  end


  tic
  for ii=1:iters
    %save previous iterate
    disp(ii)
    xprime = x;
    
    Dpv = vx+vy+vz;
    Dpv(1:N,2:N,1:M) = Dpv(1:N,2:N,1:M) - vx(1:N,1:(N-1),1:M);
    Dpv(2:N,1:N,1:M) = Dpv(2:N,1:N,1:M) - vy(1:(N-1),1:N,1:M);
    Dpv(1:N,1:N,2:M) = Dpv(1:N,1:N,2:M) - vz(1:N,1:N,1:(M-1));
    y = embed(A' * u, mask) + beta/alpha * Dpv;    
    %y = iradon(u,th,'none',N)+Dpv;
    x = x-real(ifftn(H.*fftn(y)));
    %x = x-real(ifft2(H.*fft2(iradon(u,th,'none',N)+Dpv)));
    %x = x-1/beta*y;
    
    xprime = (2*x-xprime);
    r = A * xprime(mask(:));
    %r = radon(xprime, th);
    size(u)
    size(r)
    size(sino)
 

    u = double(1/(1+alpha)*(u+alpha*(r-sino)));
    vx(1:N,1:(N-1),1:M) = max(min(vx(1:N,1:(N-1),1:M) + beta * (xprime(1:N,1:(N-1),1:M)-xprime(1:N,2:N,1:M)),lambda * alpha/beta),-lambda * alpha / beta);
    vy(1:(N-1),1:N,1:M) = max(min(vy(1:(N-1),1:N,1:M) + beta * (xprime(1:(N-1),1:N,1:M)-xprime(2:N,1:N,1:M)),lambda * alpha/beta),-lambda*alpha/beta);
    vz(1:N,1:N,1:(M-1)) = max(min(vz(1:N,1:N,1:(M-1)) + beta * (xprime(1:N,1:N,1:(M-1))-xprime(1:N,1:N,2:M)),lambda * alpha/beta),-lambda*alpha/beta);
    disp(sum(sum(sum(abs(y-mean(mean(mean(y))))))))
    if mod(ii, 10) == 0
        if ncs
            filename = sprintf('ncs_%03d.png', ii);
        else
            filename = sprintf('pdhg_%03d.png', ii);
        end
        imwrite(x(:,:,66)/1.7160e+03, filename);
    end
  end
  toc
end
