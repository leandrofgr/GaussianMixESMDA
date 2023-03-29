

close all
%% facies
dim = 2;

p1 = 0.5;
pi_ind = rand(dim,1);
pi(pi_ind<p1 ) = 1;
pi(pi_ind>=p1 ) = 2;

%% prior

lambda(1) = p1;
lambda(2) = 1 - lambda(1);

mu_m(:,1) = 0.1*ones(dim,1);
C_m(:,:,1) = 0.05^2*eye(dim);
mu_m(:,2) = 0.3*ones(dim,1);
C_m(:,:,2) = 0.05^2*eye(dim);

%% simulation of data
m_true = 0.5*rand(dim,1);
G = 10*eye(dim);

d = G*m_true;
sgm_2 = 0.5;
d = d + sgm_2*randn(dim,1);
C_d = sgm_2*eye(dim);

%% inversion by Numerical evaluation
points = 100;
min_X = 0.0;
max_X = 0.5;
min_Y = 0.0;
max_Y = 0.5;
[X,Y] = meshgrid( linspace(min_X,max_X,points ) , linspace(min_Y,max_Y,points) );

% prior
P_prior = lambda(1)*mvnpdf([X(:) Y(:)],repmat(mu_m(:,1)',points^2,1),C_m(:,:,1));
P_prior = P_prior + lambda(2)*mvnpdf([X(:) Y(:)],repmat(mu_m(:,2)',points^2,1),C_m(:,:,2));
P_prior = reshape(P_prior,points,points);

% likelyhood
Gm = (G*[X(:) Y(:)]')';
P_likely = mvnpdf(Gm,repmat(d',points^2,1),C_d);
P_likely = reshape(P_likely,points,points);

% posterior numerically
P_posterior = P_likely.*P_prior;

% inversion by GaussianMixBLI
 for facie = 1:2
     umld(:,facie) = mu_m(:,facie ) + C_m(:,:,facie)*G'*( (G*C_m(:,:,facie)*G' + C_d)\(d - G*mu_m(:,facie )) );
     Cmld(:,:,facie) = C_m(:,:,facie) - C_m(:,:,facie)*G'*( (G*C_m(:,:,facie)*G' + C_d)\(G*C_m(:,:,facie)) );
     lambda_lm(facie) = lambda(facie)*mvnpdf( d, G*mu_m(:,facie), (G*C_m(:,:,facie)*G' + C_d)  );
 end
 lambda_lm = lambda_lm/sum(lambda_lm);
 
P_posterior_analytical = lambda_lm(1)*mvnpdf([X(:) Y(:)],repmat(umld(:,1)',points^2,1),Cmld(:,:,1));
P_posterior_analytical = P_posterior_analytical+ lambda_lm(2)*mvnpdf([X(:) Y(:)],repmat(umld(:,2)',points^2,1),Cmld(:,:,2));
P_posterior_analytical = reshape(P_posterior_analytical,points,points);


 %% Plots
f = figure
f.Position = [500 500 1500 500];
subplot(1,4,1)
pcolor(X,Y,P_prior)
shading flat
title('Prior')
xlabel('m1')
ylabel('m2')

subplot(1,4,2)
pcolor(X,Y,P_likely)
shading flat
title('Likelihood')
xlabel('m1')

subplot(1,4,3)
pcolor(X,Y,P_posterior)
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
title('Posterior numerically')
xlabel('m1')

subplot(1,4,4)
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
