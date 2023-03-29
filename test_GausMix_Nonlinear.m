
% CONCLUSION: The linearized posterior is very wrong only when the
% linearization is very wrong. For example, by linearizing g in a point 
% mu_x far from the correct reference. 

close all
clear all
%% facies
dim = 2;

p1 = 0.5;
pi_ind = rand(dim,1);
pi(pi_ind<p1 ) = 1;
pi(pi_ind>=p1 ) = 2;

%% prior

lambda(1) = p1;
lambda(2) = 1 - lambda(1);

mu_m(:,1) = [0.25 ; 0.1];
C_m(:,:,1) = 0.07^2*eye(dim);
mu_m(:,2) = 0.3*ones(dim,1);
C_m(:,:,2) = 0.07^2*eye(dim);

%% simulation of data
m_true = [0.17;0.2];

% Non-linear Function
linear_factor = 0.1 ;
syms g(x)
g(x) = ( (x+linear_factor).^2 - linear_factor^2 ) / ((1+linear_factor).^2 - linear_factor^2);

% linearization
%mu_x = mean(m_true);
mu_x = 0.4 
G = eye(dim) * ( 2*( mu_x + linear_factor ) ) / ((1+linear_factor).^2 - linear_factor^2);
G_const = double(g(mu_x)) - diag(G*mu_x);

% simulation of data
sgm_2 = 2*(0.04);
d = double(g(m_true));
d = d + [0.01;-0.005];

C_d = sgm_2^2*eye(dim);


%% 1D Comparison function vs linearization
x_axis = linspace(0,1,100);
figure
subplot(1,2,1)
plot(x_axis, double(g(x_axis)) )
hold all
plot(x_axis, G(1,1)*x_axis + G_const )
xlabel('m1');ylabel('d1')
grid
subplot(1,2,2)
plot(x_axis, exp( - (sgm_2^-2) * (double(g(x_axis)) - d(1)).^2 ))
hold all
plot(x_axis, exp( - (sgm_2^-2) * (G(1,1)*x_axis + G_const - d(1)).^2 ))
xlabel('m1');ylabel('d1')
grid


%% inversion by Numerical evaluation
points = 75;
min_X = 0.0;
max_X = 0.5;
min_Y = 0.0;
max_Y = 0.5;
[X,Y] = meshgrid( linspace(min_X,max_X,points ) , linspace(min_Y,max_Y,points) );

% prior
P_prior = lambda(1)*mvnpdf([X(:) Y(:)],repmat(mu_m(:,1)',points^2,1),C_m(:,:,1));
P_prior = P_prior + lambda(2)*mvnpdf([X(:) Y(:)],repmat(mu_m(:,2)',points^2,1),C_m(:,:,2));
P_prior = reshape(P_prior,points,points);


% linear likelyhood
Gm = (G*[X(:) Y(:)]')' + G_const';
P_likely_linear = mvnpdf(Gm,repmat(d',points^2,1),C_d);
P_likely_linear = reshape(P_likely_linear,points,points);

% non linear likelyhood
Gm = double(g([X(:) Y(:)]'))';
P_likely_nonlinear = mvnpdf(Gm,repmat(d',points^2,1),C_d);
P_likely_nonlinear = reshape(P_likely_nonlinear,points,points);

% posterior numerically
P_posterior_nonlinear = P_likely_nonlinear.*P_prior;
P_posterior_linear = P_likely_linear.*P_prior;

%% inversion by GaussianMixBLI
 for facie = 1:2
     umld(:,facie) = mu_m(:,facie ) + C_m(:,:,facie)*G'*( (G*C_m(:,:,facie)*G' + C_d)\( d-G_const - G*mu_m(:,facie )) );
     Cmld(:,:,facie) = C_m(:,:,facie) - C_m(:,:,facie)*G'*( (G*C_m(:,:,facie)*G' + C_d)\(G*C_m(:,:,facie)) );
     lambda_lm(facie) = lambda(facie) * mvnpdf( d, G*mu_m(:,facie)+G_const, (G*C_m(:,:,facie)*G' + C_d)  );
 end
 lambda_lm = lambda_lm/sum(lambda_lm);
 
P_posterior_analytical = lambda_lm(1)*mvnpdf([X(:) Y(:)],repmat(umld(:,1)',points^2,1),Cmld(:,:,1));
P_posterior_analytical = P_posterior_analytical+ lambda_lm(2)*mvnpdf([X(:) Y(:)],repmat(umld(:,2)',points^2,1),Cmld(:,:,2));
P_posterior_analytical = reshape(P_posterior_analytical,points,points);



 %% Plots
f = figure
f.Position = [500 500 1500 500];
subplot(2,4,1)
pcolor(X,Y,P_prior)
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
title('Prior')
xlabel('m1')
ylabel('m2')

subplot(2,4,5)
pcolor(X,Y,P_prior)
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
title('Prior')
xlabel('m1')
ylabel('m2')

subplot(2,4,2)
pcolor(X,Y,P_likely_nonlinear)
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
title('Likelihood')
xlabel('m1')

subplot(2,4,6)
pcolor(X,Y,P_likely_linear)
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
title('Likelihood')
xlabel('m1')

subplot(2,4,3)
pcolor(X,Y,P_posterior_nonlinear)
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
title('Posterior numerically')
xlabel('m1')

subplot(2,4,7)
pcolor(X,Y,P_posterior_linear)
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
title('Posterior numerically')
xlabel('m1')

subplot(2,4,8)
pcolor(X,Y,P_posterior_analytical )
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
title('Posterior analytical ')
xlabel('m1')



%% inversion by BLI
% for facie = 1:2
%     umld(:,facie) = mu_m(:,facie ) + C_m(:,:,facie)*G'*((G*C_m(:,:,facie)*G' + C_d)\(d - G*mu_m(:,facie )));
% end
% 
% figure
% plot(m,'LineWidth',2)
% hold all
% plot(umld(:,1),'r')
% plot(umld(:,2),'k')
