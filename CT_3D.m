function x = CT_3D(N,M,  iters, lambda, alpha, beta, gamma, sino,  A, mask, ncs)

sino = reshape(sino, [], 1);

rng(999);
x = randn(N,N, M);
u = zeros(size(A, 1), 1);
vx = zeros(N,N,M);
vy = zeros(N,N,M);
vz = zeros(N,N,M);


if ~ncs
    % PDHG filter
    H =  gamma*ones(N,N,M);
    H = 1./H;
else
    %NCS filter
    [kk,ll,mm] = meshgrid(0:(N-1),0:(N-1),0:(M-1));
    H =  gamma*ones(N,N,M) + 3*alpha*ncs + 4*beta^2/alpha*((sin(kk*pi/N)).^2+(sin(ll*pi/N)).^2+(sin(mm*pi/M)).^2);
    H = 1./H;
end

for ii=1:iters
    
    %disp(ii)
    xprime = x;
    
    Dpv = vx+vy+vz;
    Dpv(1:N,2:N,1:M) = Dpv(1:N,2:N,1:M) - vx(1:N,1:(N-1),1:M);
    Dpv(2:N,1:N,1:M) = Dpv(2:N,1:N,1:M) - vy(1:(N-1),1:N,1:M);
    Dpv(1:N,1:N,2:M) = Dpv(1:N,1:N,2:M) - vz(1:N,1:N,1:(M-1));
    
    y = embed(A' * u, mask) + beta/alpha * Dpv;
    x = x-real(ifftn(H.*fftn(y)));
    %x = x-1/beta*y;
    
    
    xprime = (2*x-xprime);
    r = A * xprime(mask(:));
    
    u = double(1/(1+alpha)*(u+alpha*(r-sino)));
    vx(1:N,1:(N-1),1:M) = max(min(vx(1:N,1:(N-1),1:M) + beta*(xprime(1:N,1:(N-1),1:M)-xprime(1:N,2:N,1:M)),lambda*alpha/beta),-lambda*alpha/beta);
    vy(1:(N-1),1:N,1:M) = max(min(vy(1:(N-1),1:N,1:M) + beta*(xprime(1:(N-1),1:N,1:M)-xprime(2:N,1:N,1:M)),lambda*alpha/beta),-lambda*alpha/beta);
    vz(1:N,1:N,1:(M-1)) = max(min(vz(1:N,1:N,1:(M-1)) + beta*(xprime(1:N,1:N,1:(M-1))-xprime(1:N,1:N,2:M)),lambda*alpha/beta),-lambda*alpha/beta);

    %disp(mean(mean(mean(y))))
    
    if mod(ii, 10) == 0
        disp((1/2)*gather(sum(sum(sum((A*x(mask(:))-sino).^2)))))
        disp(lambda*gather(sum(sum(sum(abs(x(1:N,1:(N-1),1:M)-x(1:N,2:N,1:M)))))+sum(sum(sum(abs(x(1:(N-1),1:N,1:M)-x(2:N,1:N,1:M)))))+sum(sum(sum(abs(x(1:N,1:N,1:(M-1))-x(1:N,1:N,2:M)))))));
    end


    if mod(ii, 10) == 0
        disp(ii)
        if ncs
            filename_x = sprintf('ncs_%03d_x.png', ii);
            filename_y = sprintf('ncs_%03d_y.png', ii);
            filename_z = sprintf('ncs_%03d_z.png', ii);
        else
            filename_x = sprintf('pdhg_%03d_x.png', ii);
            filename_y = sprintf('pdhg_%03d_y.png', ii);
            filename_z = sprintf('pdhg_%03d_z.png', ii);
        end
        imwrite(squeeze(x(210,:,:))'/1.7e+03, filename_x);
        imwrite(squeeze(x(:,210,:))'/1.7e+03, filename_y);
        imwrite(x(:,:,66)/1.7e+03, filename_z);
    end
    
end
end
