
% CONCLUSION: The linearized posterior is very wrong only when the
% linearization is very wrong. For example, by linearizing g in a point 
% mu_x far from the correct reference. 

%% TODO:
% Colocar correlação entre m1 e m2, prior 1 com corr positiva e 2 com corr negativa

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

corr = 0.85;
mu_m(:,1) = [0.25 ; 0.1];
C_m(:,:,1) = 0.07^2*eye(2,2);
%mu_m(:,1) = [0.25 ; 0.25];
%C_m(:,:,1) = 0.07^2*[  1   corr; corr   1];

mu_m(:,2) = [0.3 ; 0.35];
C_m(:,:,2) = 0.07^2*eye(2,2);
%mu_m(:,2) = [0.25 ; 0.25];
%C_m(:,:,2) = 0.07^2*[  1   -corr; -corr   1];

% simulation of data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% simulation of data
m_true = [0.2;0.17]; 

%m_true = [0.33;0.33];




% Non-linear Function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Non-linear Function
linear_factor = 10;
sgm = 0.1;
%linear_factor = -0.6;
%sgm = 0.15;
syms g(x)
g(x) = ( (x+linear_factor).^2 - linear_factor^2 ) / ((1+linear_factor).^2 - linear_factor^2);

% linearization
%mu_x = mean(m_true);
mu_x = 0.3; 
G = eye(dim) * ( 2*( mu_x + linear_factor ) ) / ((1+linear_factor).^2 - linear_factor^2);
G_const = double(g(mu_x)) - diag(G*mu_x);

% simulation of data
% Non-linear Function
% Erro sigma %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Erro sigma
d_obs = double(g(m_true));
%d_obs = d_obs + [0.01;-0.005];

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
 lambda_bli = lambda_bli/sum(lambda_bli)
 
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

%% GaussianMix ESMDA 
n_e = 1000;
n_it = 5;

m1 = mvnrnd(mu_m(:,1),C_m(:,:,1),n_e)';
d1 = double( g(m1) );
m2 = mvnrnd(mu_m(:,2),C_m(:,:,2),n_e)';
d2 = double( g(m2) );

lambda_post = lambda;
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
    
    % FAZ SENTIDO ,  FALHA no caso de likelihood relativamente estreita, e com
    % prior com media diferentes, as probabilidade dão 0.5 justamente
    % porquelambda_esmda_plm_means m1 e m1 convergem, são sugados para a solução da likelihood
    lambda_esmda(1) = lambda(1) * mvnpdf( d_obs, double( g(mu_1_post_gibbs) ), C_dd1 + C_d  );
    lambda_esmda(2) = lambda(2) * mvnpdf( d_obs, double( g(mu_2_post_gibbs) ), C_dd2 + C_d  );
    lambda_esmda = lambda_esmda/sum(lambda_esmda);
    
    % ACHO QUE NÃO FAZ MUITO SENTIDO - O sentido é bem pratico. M1 começa
    % a deslocar em direção a likelihood e diminui essa probabilidade
    % FALHA com medias iguais e correlações diferentes
    %lambda_esmda_plm_means(:,1) = mvnpdf( mu_1_post, mu_m(:,1), C_m(:,:,1) );
    %lambda_esmda_plm_means(:,2) = mvnpdf( mu_2_post, mu_m(:,2), C_m(:,:,2) );
    %lambda_esmda_plm_means = lambda_esmda_plm_means/sum(lambda_esmda_plm_means);

    % ACHO QUE NÃO FAZ SENTIDO NENHUM - O sentido é bem pratico. M1 começa
    % a deslocar em direção a likelihood e diminui essa probabilidade
    % FALHA com medias iguais e correlações diferentes
    lambda_esmda_plm_approxProd_(:,1) = mvnpdf( m1', repmat(mu_m(:,1)',n_e,1), C_m(:,:,1) );
    lambda_esmda_plm_approxProd_(:,2) = mvnpdf( m2', repmat(mu_m(:,2)',n_e,1), C_m(:,:,2) )  ;  
    lambda_esmda_plm_approxProd = lambda_esmda_plm_approxProd_'./sum(lambda_esmda_plm_approxProd_');
    lambda_esmda_plm_approxProd = mean(lambda_esmda_plm_approxProd' );
    
    % o certo seria calcular o produto de todos os valores de lambda_esmda_plm_, mas não é possível, dá inf
    % - tem que pegar todos m e não apenas m1 e m2?    
    lambda_esmda_plm_1_approcProdAllEn_(:,1) = mvnpdf( [m1';m2'], repmat(mu_m(:,1)',2*n_e,1), C_m(:,:,1) );
    lambda_esmda_plm_1_approcProdAllEn_(:,2) = mvnpdf( [m1';m2'], repmat(mu_m(:,2)',2*n_e,1), C_m(:,:,2) )  ;  
    lambda_esmda_plm_1_approcProdAllEn = lambda_esmda_plm_1_approcProdAllEn_'./sum(lambda_esmda_plm_1_approcProdAllEn_');
    lambda_esmda_plm_1_approcProdAllEn = mean(lambda_esmda_plm_1_approcProdAllEn' );
    
    % By sampling
    lambda_esmda_plm_bySampling_(:,1) = mvnpdf( [m1';m2'], repmat(mu_m(:,1)',2*n_e,1), C_m(:,:,1) );
    lambda_esmda_plm_bySampling_(:,2) = mvnpdf( [m1';m2'], repmat(mu_m(:,2)',2*n_e,1), C_m(:,:,2) )  ;  
    lambda_esmda_plm_bySampling_= (lambda_esmda_plm_bySampling_'./sum(lambda_esmda_plm_bySampling_'))';
       
    r_sort= rand(2*n_e,1);    
    lambda_esmda_plm_bySampling(1) = sum(r_sort < lambda_esmda_plm_bySampling_(:,1))/(2*n_e);
    lambda_esmda_plm_bySampling(2) = 1 - lambda_esmda_plm_bySampling(1);
    

    
end

mu_1_post_gm_esmda = mean(m1')';
mu_2_post_gm_esmda = mean(m2')';

lambda_esmda 
%lambda_esmda_plm_means
lambda_esmda_plm_approxProd
lambda_esmda_plm_1_approcProdAllEn
lambda_esmda_plm_bySampling

P_posterior_esmda_gm = lambda_esmda(1)*ksdensity(m1',[X(:),Y(:)]) + lambda_esmda(2)*ksdensity(m2',[X(:),Y(:)]);
%P_posterior_esmda_gm = lambda_esmda_plm_means(1)*ksdensity(m1',[X(:),Y(:)]) + lambda_esmda_plm_means(2)*ksdensity(m2',[X(:),Y(:)]);
%P_posterior_esmda_gm = lambda_esmda_plm_approxProd(1)*ksdensity(m1',[X(:),Y(:)]) + lambda_esmda_plm_approxProd(2)*ksdensity(m2',[X(:),Y(:)]);
%P_posterior_esmda_gm = lambda_esmda_plm_1_approcProdAllEn(1)*ksdensity(m1',[X(:),Y(:)]) + lambda_esmda_plm_1_approcProdAllEn(2)*ksdensity(m2',[X(:),Y(:)]);
%P_posterior_esmda_gm = lambda_esmda_plm_bySampling(1)*ksdensity(m1',[X(:),Y(:)]) + lambda_esmda_plm_bySampling(2)*ksdensity(m2',[X(:),Y(:)]);

P_posterior_esmda_gm = reshape(P_posterior_esmda_gm,points,points);


%% GaussianMix ESMDA - Gibbs Approach
n_e = 1000;
n_it = 5;

m1 = mvnrnd(mu_m(:,1),C_m(:,:,1),n_e)';
n_e1 = size(m1,2);  
m2 = mvnrnd(mu_m(:,2),C_m(:,:,2),n_e)';
n_e2 = size(m2,2);  

lambda_post = lambda;
for it = 1:n_it
    
    d1 = double( g(m1) );
    d_per = d_obs + sqrt(n_it) * sgm * randn(dim,n_e1);
    
    Am1 = m1 - mean(m1')';    
    Ad1 = d1 - mean(d1')';
    C_md1 = Am1*Ad1'/(n_e1 -1);
    C_dd1 = Ad1*Ad1'/(n_e1 -1);    
    m1 = m1 + C_md1 * inv( C_dd1 + (n_it) * C_d ) * ( d_per - d1 ) ;    
    mu_1_post_gibbs = mean(m1')';
    
    
    d2 = double( g(m2) );
    d_per = d_obs + sqrt(n_it) * sgm * randn(dim,n_e2);
    
    Am2 = m2 - mean(m2')';
    Ad2 = d2 - mean(d2')';
    C_md2 = Am2*Ad2'/(n_e2 -1);
    C_dd2 = Ad2*Ad2'/(n_e2 -1);    
    m2 = m2 + C_md2 * inv( C_dd2 + (n_it) * C_d ) * ( d_per - d2 ) ;        
    mu_2_post_gibbs = mean(m2')';              
    
    m = [m1';m2'];
    lambda_esmda_plm_1(:,1) = mvnpdf( m, repmat(mu_m(:,1)',2*n_e,1), C_m(:,:,1) );
    lambda_esmda_plm_1(:,2) = mvnpdf( m, repmat(mu_m(:,2)',2*n_e,1), C_m(:,:,2) )  ;  
    lambda_esmda_plm1 = lambda_esmda_plm_1'./sum(lambda_esmda_plm_1');     
    
    %%%%%%% TO THINK IN A BETTER CONTROL IN CASE ONE OF THE ENSEMBLE COLAPSE/VANISH
    %lambda_esmda_plm1(lambda_esmda_plm1<0.1) = 0.11;
    %lambda_esmda_plm1 = lambda_esmda_plm_1'./sum(lambda_esmda_plm_1');     
    %%%%%%%
    
    pi = zeros(2*n_e,1);
    r_sort= rand(2*n_e,1);    
    pi(r_sort<lambda_esmda_plm1(1,:)' ) = 1;
    pi(pi==0) = 2;    
    
    m1 = m(pi == 1,:)';
    m2 = m(pi == 2,:)';
    
    n_e1 = size(m1,2);  
    n_e2 = size(m2,2);     
    
    lambda_esmda_gibbs(1) = n_e1/(2*n_e);
    lambda_esmda_gibbs(2) = n_e2/(2*n_e);
    
end

lambda_esmda_gibbs

m = [m1';m2'];
P_posterior_esmda_gm_gibbs = ksdensity(m,[X(:),Y(:)]); 
P_posterior_esmda_gm_gibbs = reshape(P_posterior_esmda_gm_gibbs,points,points);


%% GaussianMix ENKF - Dovera approach

clear m
n_e = 1000;


lambda_esmda(1) = lambda(1) * mvnpdf( d_obs, double( g(mu_m(:,1)) ), C_dd1 + C_d  );
lambda_esmda(2) = lambda(2) * mvnpdf( d_obs, double( g(mu_m(:,2)) ), C_dd2 + C_d  );
lambda_esmda = lambda_esmda/sum(lambda_esmda);

L(:,:,1) = chol(C_m(:,:,1))';
L(:,:,2) = chol(C_m(:,:,2))';

for f = 1:2*n_e
    
    k(f) = 1;
    k_sort = rand;
    if k_sort >  lambda(1) 
        k(f) = 2;
    end

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
    
    l_(f) = 1;
    l_sort = rand;
    if l_sort >  lambda_esmda(1) 
        l_(f) = 2;
    end
        
    m(:,f) = mu_m(:,l_(f)) + L(:,:,l_(f))*inv( L(:,:,k(f)) ) * ( m(:,f) - mu_m(:,k(f)) );
    
    m(:,f) = m(:,f) + C_md(:,:,l_(f)) * inv( C_dd(:,:,l_(f)) + C_d ) * ( d_obs - double(g(m(:,f))) ) ;         
        
end

m1 = m(:,l_==1);
m2 = m(:,l_==2);
mu_1_post_gm_enkf = mean(m1')';
mu_2_post_gm_enkf = mean(m2')';

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
    
    k(f) = 1;
    k_sort = rand;
    if k_sort >  lambda(1) 
        k(f) = 2;
    end

    m(:,f) = mu_m_(:,k(f)) + L(:,:,k(f)) * randn(size(m_true));     
    
end

m1 = m(:,k==1);
m2 = m(:,k==2);

for it = 1:n_it

    lambda_esmda(1) = lambda(1) * mvnpdf( d_obs, double( g(mu_m_(:,1)) ), C_dd1 + C_d  );
    lambda_esmda(2) = lambda(2) * mvnpdf( d_obs, double( g(mu_m_(:,2)) ), C_dd2 + C_d  );
    lambda_esmda = lambda_esmda/sum(lambda_esmda);
    
    L(:,:,1) = chol(C_m_(:,:,1))';
    L(:,:,2) = chol(C_m_(:,:,2))';
    
    for f = 1:size(m,2)
        
        l_(f) = 1;
        l_sort = rand;
        if l_sort >  lambda_esmda(1)
            l_(f) = 2;
        end
        
        m(:,f) = mu_m_(:,l_(f)) + L(:,:,l_(f))*inv( L(:,:,k(f)) ) * ( m(:,f) - mu_m_(:,k(f)) );
        
        %m(:,f) = m(:,f) + C_md(:,:,l_(f)) * inv( C_dd(:,:,l_(f)) + C_d ) * ( d_obs - double(g(m(:,f))) ) ;
        
    end
    
    k = l_;
    
    m1 = m(:,l_==1);
    m2 = m(:,l_==2);
    
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
    
    mu_m_(:,1) = mean(m1')';
    C_m_(:,:,1) = cov(m1')';
    mu_m_(:,2) = mean(m2')';
    C_m_(:,:,2) = cov(m2')';
    
end

mu_1_post_gm_esmda_dov = mean(m1')';
mu_2_post_gm_esmda_dov = mean(m2')';


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
xlabel('m1')


subplot(3,4,8)
pcolor(X,Y,P_posterior_linear)
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
title('Posterior numerically (linearized)')
xlabel('m1')



%%%%%    ESMDA    %%%%%

subplot(3,4,9)
pcolor(X,Y,P_posterior_gm_enkf)
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
scatter(mu_1_post_gm_enkf(1),mu_1_post_gm_enkf(2),20,'w','filled')
scatter(mu_2_post_gm_enkf(1),mu_2_post_gm_enkf(2),20,'w','filled')
title('GM ENKF (Dovera)')
xlabel('m1')

subplot(3,4,10)
pcolor(X,Y,P_posterior_esmda_gm_Dovera)
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
scatter(mu_1_post_gm_esmda_dov(1),mu_1_post_gm_esmda_dov(2),20,'w','filled')
scatter(mu_2_post_gm_esmda_dov(1),mu_2_post_gm_esmda_dov(2),20,'w','filled')
title('GM ESMDA (Dovera)')
xlabel('m1')

subplot(3,4,11)
pcolor(X,Y,P_posterior_esmda_gm)
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
scatter(mu_1_post_gm_esmda(1),mu_1_post_gm_esmda(2),20,'w','filled')
scatter(mu_2_post_gm_esmda(1),mu_2_post_gm_esmda(2),20,'w','filled')
title('GM ESMDA')
xlabel('m1')

subplot(3,4,12)
pcolor(X,Y,P_posterior_esmda_gm_gibbs )
shading flat
hold all 
scatter(m_true(1),m_true(2),20,'r','filled')
scatter(mu_1_post_gibbs(1),mu_1_post_gibbs(2),20,'w','filled')
scatter(mu_2_post_gibbs(1),mu_2_post_gibbs(2),20,'w','filled')
title('GM ESMDA Gibbs')
xlabel('m1')




