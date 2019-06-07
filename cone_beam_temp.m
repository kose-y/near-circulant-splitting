


%Code for generating true sinogram
load('../PWLS-ULTRA-for-Low-Dose-3D-CT-Image-Reconstruction/data/3Dxcat/phantom_crop154.mat');  % unit of the loaded phantom: HU
phantom = phantom(:,:,1:96); % extract 96 slices for testing

fprintf('generating noiseless sino...\n');
down = 1; % downsample rate
ig_hi = image_geom('nx',840,'dx',500/1024,'nz',96,'dz',0.625,'down',down);
down = 8; % downsample rate
cg = ct_geom('ge2', 'down', down);
A_hi = Gcone(cg, ig_hi, 'type', 'sf2', 'nthread', jf('ncore')*2-1);  
sino_true = A_hi * phantom;  clear A_hi;

save('3D_sino_true.mat', 'sino_true')
%}




load('3D_sino_true.mat')
down = 1;
ig = image_geom('nx',420,'dx',500/512,'nz',96,'dz',0.625,'down',down);
ig.mask = ig.circ > 0; % can be omitted

down = 8; % downsample rate
cg = ct_geom('ge2', 'down', down);

A = Gcone(cg, ig, 'type', 'sf2', 'nthread', jf('ncore')*2-1);



%{
%Code for generating preconditioning matrix
H = zeros(420,420,96);
count = 0;
tic
for ii=1:42:420
    for jj=1:42:420
        for kk=1:8:96
            disp([num2str(ii),num2str(jj),num2str(kk)])
            ek = zeros(420,420,96);
            ek(ii,jj,kk) = 1;
            H = H + fftn(embed(A' * (A * ek(ig.mask(:))), ig.mask))./fftn(ek);
            count = count + 1;
        end
    end
end
toc


H = real(H)/count;
save('H_mask.mat', 'H')
%}


%load('H_mask.mat', 'H')


ek = zeros(420,420,96);
ek(420/2,420/2,96/2) = 1;
H = real(fftn(embed(A' * (A * ek(ig.mask(:))), ig.mask))./fftn(ek));


%x_ncs = CT_3D(420, 96, 200, 1, 0.001, 0.1, 0.001, sino_true, A, ig.mask, H);

x_pdhg = CT_3D(420, 96, 200, 1, 0.001, 0.1, 0.3, sino_true, A, ig.mask, false);

x_pdhg = CT_3D(420, 96, 200, 1, 0.001, 0.1, 1, sino_true, A, ig.mask, false);

x_pdhg = CT_3D(420, 96, 200, 1, 0.001, 0.1, 3, sino_true, A, ig.mask, false);


x_pdhg = CT_3D(420, 96, 200, 1, 0.001, 0.1, 10, sino_true, A, ig.mask, false);




%}
