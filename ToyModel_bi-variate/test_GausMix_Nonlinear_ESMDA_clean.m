
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

lambda(1) = 0.5;
lambda(2) = 1 - lambda(1);

%% Experiment 1
% Priors with the same variances, same correlation but different means
mu_m(:,1) = [0.25 ; 0.1];
C_m(:,:,1) = 0.07^2*eye(2,2);

mu_m(:,2) = [0.3 ; 0.35];
C_m(:,:,2) = 0.07^2*eye(2,2);

m_true = [0.2;0.17]; 

%% Experiment 2
%Priors with the same means and variances, but opposite correlations
% corr = 0.85;
% 
% mu_m(:,1) = [0.25 ; 0.25];
% C_m(:,:,1) = 0.07^2*[  1   corr; corr   1];
% mu_m(:,2) = [0.25 ; 0.25];
% C_m(:,:,2) = 0.07^2*[  1   -corr; -corr   1];
% 
% m_true = [0.33;0.33];


%% Simulation of data

% PARAMETER - Non-linear Function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Non-linear Function
linear_factor = 0.4;
syms g(x)
g(x) = ( (x+linear_factor).^2 - linear_factor^2 ) / ((1+linear_factor).^2 - linear_factor^2);

% linearization
%mu_x = mean(m_true);
mu_x = 0.3; 
G = eye(dim) * ( 2*( mu_x + linear_factor ) ) / ((1+linear_factor).^2 - linear_factor^2);
G_const = double(g(mu_x)) - diag(G*mu_x);

d_obs = double(g(m_true));

% PARAMETER - Signal to Noise %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     Signal to Noise
signal2noise = 2;
%signal2noise = 5;
%signal2noise = 10;

sgm = sqrt(mean(d_obs.^2) / signal2noise );
C_d = sgm^2*eye(dim);


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
plot(x_axis, exp( - (sgm^-2) * (double(g(x_axis)) - d_obs(1)).^2 ))
hold all
plot(x_axis, exp( - (sgm^-2) * (G(1,1)*x_axis + G_const - d_obs(1)).^2 ))
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
P_likely_linear = mvnpdf(Gm,repmat(d_obs',points^2,1),C_d);
P_likely_linear = reshape(P_likely_linear,points,points);

% non linear likelyhood
Gm = double(g([X(:) Y(:)]'))';
P_likely_nonlinear = mvnpdf(Gm,repmat(d_obs',points^2,1),C_d);
P_likely_nonlinear = reshape(P_likely_nonlinear,points,points);

% posterior numerically
P_posterior_nonlinear = P_likely_nonlinear.*P_prior;
P_posterior_linear = P_likely_linear.*P_prior;

%% inversion by GaussianMixBLI
 for facie = 1:2
     umld(:,facie) = mu_m(:,facie ) + C_m(:,:,facie)*G'*( (G*C_m(:,:,facie)*G' + C_d)\( d_obs-G_const - G*mu_m(:,facie )) );
     Cmld(:,:,facie) = C_m(:,:,facie) - C_m(:,:,facie)*G'*( (G*C_m(:,:,facie)*G' + C_d)\(G*C_m(:,:,facie)) );
     lambda_bli(facie) = lambda(facie) * mvnpdf( d_obs, G*mu_m(:,facie)+G_const, (G*C_m(:,:,facie)*G' + C_d)  );
 end
 lambda_bli = lambda_bli/sum(lambda_bli);
 
P_posterior_analytical_linearized = lambda_bli(1)*mvnpdf([X(:) Y(:)],repmat(umld(:,1)',points^2,1),Cmld(:,:,1));
P_posterior_analytical_linearized = P_posterior_analytical_linearized+ lambda_bli(2)*mvnpdf([X(:) Y(:)],repmat(umld(:,2)',points^2,1),Cmld(:,:,2));
P_posterior_analytical_linearized = reshape(P_posterior_analytical_linearized,points,points);

%% ESMDA Unimodal
n_e = 1000;
n_it = 5;
prior_index = 2;
m = mvnrnd(mu_m(:,prior_index),C_m(:,:,prior_index),n_e)';
d = double( g(m) );

for it = 1:n_it
    d_per = d_obs + sqrt(n_it) * sgm * randn(dim,n_e);
    Am = m - mean(m')';
    Ad = d - mean(d')';
    C_md = Am*Ad'/(n_e-1);
    C_dd = Ad*Ad'/(n_e-1);
    m = m + C_md * inv( C_dd + sqrt(n_it) * C_d ) * ( d_per - d ) ;
    d = double( g(m) );
end

P_posterior_esmda = ksdensity(m',[X(:),Y(:)]);
P_posterior_esmda = reshape(P_posterior_esmda,points,points);

mu_post_esmda = mean( m' )';
C_post_esmda = cov( m' )';

P_prior_esmda = lambda(prior_index)*mvnpdf([X(:) Y(:)],repmat(mu_m(:,prior_index)',points^2,1),C_m(:,:,prior_index));
P_prior_esmda = reshape(P_prior_esmda,points,points);

%% GaussianMix ESMDA - Simplest approach
n_e = 1000;
n_it = 5;

m1 = mvnrnd(mu_m(:,1),C_m(:,:,1),n_e)';
d1 = double( g(m1) );
m2 = mvnrnd(mu_m(:,2),C_m(:,:,2),n_e)';
d2 = double( g(m2) );

lambda_esmda = lambda;
%lambda_post = lambda;
for it = 1:n_it
    
    d_per = d_obs + sqrt(n_it) * sgm * randn(dim,n_e);
    
    Am1 = m1 - mean(m1')';
    Ad1 = d1 - mean(d1')';
    C_md1 = Am1*Ad1'/(n_e-1);
    C_dd1 = Ad1*Ad1'/(n_e-1);    
    m1 = m1 + C_md1 * inv( C_dd1 + (n_it) * C_d ) * ( d_per - d1 ) ;
    d1 = double( g(m1) );
    mu_1_post_gibbs = mean(m1')';
    
    Am2 = m2 - mean(m2')';
    Ad2 = d2 - mean(d2')';
    C_md2 = Am2*Ad2'/(n_e-1);
    C_dd2 = Ad2*Ad2'/(n_e-1);    
    m2 = m2 + C_md2 * inv( C_dd2 + (n_it) * C_d ) * ( d_per - d2 ) ;
    d2 = double( g(m2) );
    mu_2_post_gibbs = mean(m2')';
        
    lambda_esmda(1) = lambda_esmda(1) * mvnpdf( d_obs, double( g(mu_1_post_gibbs) ), C_dd1 + C_d  );
    lambda_esmda(2) = lambda_esmda(2) * mvnpdf( d_obs, double( g(mu_2_post_gibbs) ), C_dd2 + C_d  );
    lambda_esmda = lambda_esmda/sum(lambda_esmda);
      
end

mu_1_post_gm_esmda = mean(m1')';
mu_2_post_gm_esmda = mean(m2')';


P_posterior_esmda_gm = lambda_esmda(1)*ksdensity(m1',[X(:),Y(:)]) + lambda_esmda(2)*ksdensity(m2',[X(:),Y(:)]);
P_posterior_esmda_gm = reshape(P_posterior_esmda_gm,points,points);


%% GaussianMix ENKF - Dovera approach

clear m
n_e = 1000;

lambda_esmda(1) = lambda(1) * mvnpdf( d_obs, double( g(mu_m(:,1)) ), C_dd1 + C_d  );
lambda_esmda(2) = lambda(2) * mvnpdf( d_obs, double( g(mu_m(:,2)) ), C_dd2 + C_d  );
lambda_esmda = lambda_esmda/sum(lambda_esmda);

L(:,:,1) = chol(C_m(:,:,1))';
L(:,:,2) = chol(C_m(:,:,2))';

for f = 1:2*n_e
    
    k(f) = (rand <= lambda(2) ) + 1;       

    m(:,f) = mu_m(:,k(f)) + L(:,:,k(f)) * randn(size(m_true));     
    
end

m1 = m(:,k==1);
m2 = m(:,k==2);

n_e1 = size(m1,2);
n_e2 = size(m2,2);

d1 = double( g(m1) );

Am1 = m1 - mean(m1')';
Ad1 = d1 - mean(d1')';
C_md1 = Am1*Ad1'/(n_e1 -1);
C_dd1 = Ad1*Ad1'/(n_e1 -1);

d2 = double( g(m2) );

Am2 = m2 - mean(m2')';
Ad2 = d2 - mean(d2')';
C_md2 = Am2*Ad2'/(n_e2 -1);
C_dd2 = Ad2*Ad2'/(n_e2 -1);

C_dd(:,:,1) = C_dd1;
C_dd(:,:,2) = C_dd2;

C_md(:,:,1) = C_md1;
C_md(:,:,2) = C_md2;


for f = 1:size(m,2)
    
    l_(f) = (rand <= lambda_esmda(2) ) + 1;               
        
    m(:,f) = mu_m(:,l_(f)) + L(:,:,l_(f))*inv( L(:,:,k(f)) ) * ( m(:,f) - mu_m(:,k(f)) );
    
    m(:,f) = m(:,f) + C_md(:,:,l_(f)) * inv( C_dd(:,:,l_(f)) + C_d ) * ( d_obs - double(g(m(:,f))) ) ;         
        
end

m1 = m(:,l_==1);
m2 = m(:,l_==2);
mu_1_post_gm_enkf = mean(m1')';
mu_2_post_gm_enkf = mean(m2')';

n_e1 = size(m1,2);  
n_e2 = size(m2,2);         
lambda_esmda_gm_enkf(1) = n_e1/(2*n_e);
lambda_esmda_gm_enkf(2) = n_e2/(2*n_e);


P_posterior_gm_enkf = ksdensity(m',[X(:),Y(:)]);
P_posterior_gm_enkf  = reshape(P_posterior_gm_enkf,points,points);


%% GaussianMix ESMDA - Inspired by GM ENKF Dovera)

n_e = 1000;
n_it = 5;
clear m

lambda_esmda = lambda;

% Prior sampling
C_m_ = C_m;
mu_m_ = mu_m;

for f = 1:2*n_e
    
    k(f) = (rand <= lambda(2) ) + 1;       
    m(:,f) = mu_m_(:,k(f)) + L(:,:,k(f)) * randn(size(m_true));     
    
end

m1 = m(:,k==1);
m2 = m(:,k==2);

for it = 1:n_it
    
    L(:,:,1) = chol(C_m_(:,:,1))';
    L(:,:,2) = chol(C_m_(:,:,2))';
    
    n_e1 = size(m1,2);
    n_e2 = size(m2,2);
    
    d1 = double( g(m1) );
    d_per1 = d_obs + sqrt(n_it) * sgm * randn(dim,n_e1);
    
    Am1 = m1 - mean(m1')';
    Ad1 = d1 - mean(d1')';
    C_md1 = Am1*Ad1'/(n_e1 -1);
    C_dd1 = Ad1*Ad1'/(n_e1 -1);
    
    d2 = double( g(m2) );
    d_per2 = d_obs + sqrt(n_it) * sgm * randn(dim,n_e2);
    
    Am2 = m2 - mean(m2')';
    Ad2 = d2 - mean(d2')';
    C_md2 = Am2*Ad2'/(n_e2 -1);
    C_dd2 = Ad2*Ad2'/(n_e2 -1);
    
    C_dd(:,:,1) = C_dd1;
    C_dd(:,:,2) = C_dd2;
    
    C_md(:,:,1) = C_md1;
    C_md(:,:,2) = C_md2;        
    
    m1 = m1 + C_md1 * inv( C_dd1 + (n_it) * C_d ) * ( d_per1 - d1 ) ;    
    m2 = m2 + C_md2 * inv( C_dd2 + (n_it) * C_d ) * ( d_per2 - d2 ) ;        
    
    lambda_esmda(1) = lambda(1) * mvnpdf( d_obs, double( g(mu_m_(:,1)) ), C_dd1 + C_d  );
    lambda_esmda(2) = lambda(2) * mvnpdf( d_obs, double( g(mu_m_(:,2)) ), C_dd2 + C_d  );
    lambda_esmda = lambda_esmda/sum(lambda_esmda);    
       
    
    for f = 1:size(m,2)
                
        l_(f) = (rand <= lambda_esmda(2) ) + 1;               
        m(:,f) = mu_m_(:,l_(f)) + L(:,:,l_(f))*inv( L(:,:,k(f)) ) * ( m(:,f) - mu_m_(:,k(f)) );
                
    end
    
    k = l_;    
        
    m1 = m(:,l_==1);
    m2 = m(:,l_==2);    
        
end

mu_1_post_gm_esmda_dov = mean(m1')';
mu_2_post_gm_esmda_dov = mean(m2')';

n_e1 = size(m1,2);  
n_e2 = size(m2,2);         
lambda_esmda_esmda_gm_Dovera(1) = n_e1/(2*n_e);
lambda_esmda_esmda_gm_Dovera(2) = n_e2/(2*n_e);


m = [m1';m2'];
P_posterior_esmda_gm_Dovera = ksdensity(m,[X(:),Y(:)]);
P_posterior_esmda_gm_Dovera = reshape(P_posterior_esmda_gm_Dovera,points,points);



 %% Plots
f = figure;
f.Position = [500 100 1500 900];

%%%%%    NON LINEAR GENERAL    %%%%%

subplot(3,4,1)
pcolor(X,Y,P_prior)
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
title('Prior')
xlabel('m1')
ylabel('m2')

subplot(3,4,2)
pcolor(X,Y,P_likely_nonlinear)
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
title('Likelihood')
xlabel('m1')

subplot(3,4,3)
pcolor(X,Y,P_posterior_nonlinear)
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
title('Posterior numerically')
xlabel('m1')

%%%%%    LINEARIZED    %%%%%

subplot(3,4,5)
pcolor(X,Y,P_prior)
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
title('Prior')
xlabel('m1')
ylabel('m2')

subplot(3,4,6)
pcolor(X,Y,P_likely_linear)
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
title('Likelihood (linearized)')
xlabel('m1')

subplot(3,4,7)
pcolor(X,Y,P_posterior_analytical_linearized )
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
scatter(umld(1,1),umld(2,1),20,'w','filled')
scatter(umld(1,2),umld(2,2),20,'w','filled')
title('Posterior analytical (linearized) ')
xlabel(lambda_bli(1))


subplot(3,4,8)
pcolor(X,Y,P_posterior_linear)
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
title('Posterior numerically (linearized)')
xlabel('m1')


%%%%%    ESMDA    %%%%%

subplot(3,4,10)
pcolor(X,Y,P_posterior_esmda_gm)
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
scatter(mu_1_post_gm_esmda(1),mu_1_post_gm_esmda(2),20,'w','filled')
scatter(mu_2_post_gm_esmda(1),mu_2_post_gm_esmda(2),20,'w','filled')
title('GM ESMDA')
xlabel(lambda_esmda(1))

subplot(3,4,11)
pcolor(X,Y,P_posterior_gm_enkf)
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
scatter(mu_1_post_gm_enkf(1),mu_1_post_gm_enkf(2),20,'w','filled')
scatter(mu_2_post_gm_enkf(1),mu_2_post_gm_enkf(2),20,'w','filled')
title('GM ENKF (Dovera)')
xlabel(lambda_esmda_gm_enkf(1))

subplot(3,4,12)
pcolor(X,Y,P_posterior_esmda_gm_Dovera)
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
scatter(mu_1_post_gm_esmda_dov(1),mu_1_post_gm_esmda_dov(2),20,'w','filled')
scatter(mu_2_post_gm_esmda_dov(1),mu_2_post_gm_esmda_dov(2),20,'w','filled')
title('GM ESMDA (Dovera)')
xlabel(lambda_esmda_esmda_gm_Dovera(1))





