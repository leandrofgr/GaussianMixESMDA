
close all
clear

addpath('Functions')

%% LOAD FACIES MODEL, MODIFY ITS SIZE AND 
facies = load('facies_model.mat','facies').facies;
facies = facies(20:40,72:92);
facies = imresize(facies,1);
facies(facies>1.5) = 2;

facies(facies<=1.5) = 1;

I = size(facies,1);
J = size(facies,2);

%% PRIOR DISTRIBUTIONS
lambda(1) = 0.5;
lambda(2) = 1 - lambda(1);

C_m_matrix = covariance_matrix_exp(0.02^2*ones(I,1),3,1);
C_m_matrix = repmat(C_m_matrix,J,J);

mu_m(1) = 0.1;
C_m(1) = 0.02^2;
mu_m(2) = 0.2;
C_m(2) = 0.02^2;


%% SIMULATE TRUE POROSITY MODEL
[correlation_function1] = construct_correlation_function_beta(3, 3, facies, 2);
[correlation_function2] = construct_correlation_function_beta(1, 3, facies, 2);
[ simulation1 ] = mu_m(1) + sqrt(C_m(1)) * FFT_MA_3D( correlation_function1, randn(I,J));
[ simulation2 ] = mu_m(2) + sqrt(C_m(2)) * FFT_MA_3D( correlation_function2, randn(I,J));

porosity(facies == 1) = simulation1(facies == 1);
porosity(facies == 2) = simulation2(facies == 2);
porosity = reshape(porosity, I,J);


%% SIMULATE OBSERVATIONS AND CONSTRUCT THE FORWARD MODEL]

n_wells_per_dim = 4;

positions = linspace(1,I,n_wells_per_dim + 2);
positions = round(positions(2:end-1));
[X_positions, Y_positions] = meshgrid(positions ,positions);

for well = 1:n_wells_per_dim^2
    aux_position = zeros(I,J);
    aux_position(X_positions(well),Y_positions(well)) = 1;
    G(well,:) = aux_position(:);
end


d_obs = G * porosity(:);

sgm = 0.0001;
C_d = sgm*ones(size(d_obs,1),size(d_obs,1));

figure
subplot(121)
imagesc(facies)
hold all
scatter(X_positions,Y_positions,15,'r')
subplot(122)
imagesc(porosity)
hold all
scatter(X_positions,Y_positions,15,'r')

%% ESMDA UNIMODAL
 
n_e = 200;
n_it = 5;
prior_index = 2;

for f = 1:n_e
    [ m(:,f) ] = reshape( mean(mu_m) + 1*sqrt(C_m(1)) * FFT_MA_3D( correlation_function1, randn(I,J)),I*J,1);
end

d = G*m;

for it = 1:n_it
    d_per = d_obs + sqrt(n_it) * sgm * randn(size(d_obs,1),n_e);
    Am = m - mean(m')';
    Ad = d - mean(d')';
    C_md = Am*Ad'/(n_e-1);
    C_dd = Ad*Ad'/(n_e-1);
    m = m + C_md * inv( C_dd + sqrt(n_it) * C_d ) * ( d_per - d ) ;
    d = double( G*m );
end
 
m_mean_esmda_unimodal = reshape(mean(m,2),I,J);

figure
subplot(131)
imagesc(facies)
hold all
scatter(X_positions,Y_positions,15,'r')
title('Facies ref model')
subplot(132)
imagesc(porosity)
hold all
scatter(X_positions,Y_positions,15,'r')
title('Porosity ref model')
subplot(133)
imagesc(m_mean_esmda_unimodal)
hold all
scatter(X_positions,Y_positions,15,'r')
title('ESMDA Unimodal')


%% GaussianMix ESMDA - Inspired by GM ENKF Dovera)
% Duvida: Na Dovera, a equação de update dos modelos usa G e Cm|pi (depende da facies pi)
% No caso do ESMDA o filtro de kalman/update é calculado baseado no
% ensemble. Portanto, para o update do ESMDA, precisa-se de uma quantidade
% razoavel de modelos referentes a cada uma das configurações possíveis de
% fácies. Com objetivo de calcular as matrizes de covariance baseado numa
% estatística,
% Durante a exeução do Dovera, a quantidade de configurações de fácies é
% enorme, muito maior que a quantidade de modelos do ensemble. O que
% inviabiliza a aplicação de tal abordagem no ESMDA. 

% Solução 1: Tentar pixel a pixel
% Solução 2: Tentar um esquema de importance sampling. No lugar de
% trabalhar com todas as possíveis combinações de facies, trabalha apenas
% com uma quantidade inicialmente amostrada da prior. 

n_e = 1000;
n_it = 5;

window_size = 3;
n_facies = 2^(3^2);

P_hor = [0.75 0.25;
         0.25 0.75];
P_ver = P_hor;


% ALL POSSIBLE FACIES CONFIGS FROM CHAT GPT
configurations = dec2bin(0:2^(window_size * window_size )-1)-'0';
configurations = reshape(configurations,[],window_size,window_size );
configurations(configurations==0) = 2;
configurations = flipud(configurations );

% Defining Prior means
mu_m_ = zeros(size(configurations));
mu_m_(configurations==1) = mu_m(1);
mu_m_(configurations==2) = mu_m(2);

% Defining Prior covs (assuming the same for both facies for simplicity)
L_corr = 2;
C_m_matrix_window = covariance_matrix_exp(C_m(1)*ones(window_size,1),L_corr,1);
C_m_matrix_window = repmat(C_m_matrix_window,window_size,window_size);
L = chol(C_m_matrix_window + 0.01*mean(diag(C_m_matrix_window))*eye(size(C_m_matrix_window)) )';
    
for f = 1:n_e
    facies_config(1,:,:) = simulate_markov_2Dchain(P_hor,P_ver,ones(window_size,window_size,2));
    diffe = reshape(configurations - repmat(facies_config,n_facies,1),n_facies ,9);
    k(f) = find(sum(abs(diffe),2)==0);
    
    media = mu_m_(k(f),:,:);
    %m(:,f) = media(:) + L * randn(size(media));
    
end

%for facies_config = 1:




%% EVALUATION OF GM BLI WEIGHTS

P_hor = [0.75 0.25;
         0.25 0.75];
P_ver = P_hor;

window_size = 3;

if I == window_size 
    
% ALL POSSIBLE FACIES CONFIGS FROM CHAT GPT
configurations = dec2bin(0:2^(window_size * window_size )-1)-'0';
configurations = reshape(configurations,[],window_size,window_size );
configurations(configurations==0) = 2;
configurations = flipud(configurations );

means = zeros(size(configurations));
means(configurations==1) = mu_m(1);
means(configurations==2) = mu_m(2);
 
facies_prob = zeros(I,J);
for config = 1:size(configurations,1)
    mulpi = means(config,:,:);
    prob(config) = compute_Markov_prior(squeeze(configurations(config,:,:)),P_hor,P_ver);
    prob(config) = prob(config) * mvnpdf( d_obs, G*mulpi(:), G*C_m_matrix*G' + C_d );
    facies_prob = facies_prob + prob(config) * squeeze( configurations(config,:,:) ) ;
end

figure 
plot(prob)
figure
imagesc(facies_prob)

end


