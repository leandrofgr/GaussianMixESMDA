
close all
clear

addpath('Functions')

%% LOAD FACIES MODEL, MODIFY ITS SIZE AND 
facies = load('facies_model.mat','facies').facies;
% facies = facies(15:45,24:54);
facies = facies(22:42,70:90);
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
[correlation_function1] = construct_correlation_function_beta(3, 3, zeros(4*I,4*J), 2);
[correlation_function2] = construct_correlation_function_beta(1, 3, zeros(4*I,4*J), 2);
[ simulation1 ] = mu_m(1) + sqrt(C_m(1)) * FFT_MA_3D( correlation_function1, randn(4*I,4*J));
simulation1 = simulation1 (1:I, 1:J);
[ simulation2 ] = mu_m(2) + sqrt(C_m(2)) * FFT_MA_3D( correlation_function2, randn(4*I,4*J));
simulation2 = simulation2 (1:I, 1:J);

porosity(facies == 1) = simulation1(facies == 1);
porosity(facies == 2) = simulation2(facies == 2);
porosity = reshape(porosity, I,J);


%% SIMULATE OBSERVATIONS AND CONSTRUCT THE FORWARD MODEL

n_wells_per_dim = 3;
n_index_around_well = 4;

positions = linspace(1,I,n_wells_per_dim + 2);
positions = round(positions(2:end-1));
[X_positions, Y_positions] = meshgrid(positions ,positions);

%Mesaure 1 - Value at well location
for well = 1:n_wells_per_dim^2
    [X_positions, Y_positions] = meshgrid(positions ,positions);
    aux_position = zeros(I,J);
    aux_position(X_positions(well),Y_positions(well)) = 1;
    G_(well,:) = aux_position(:);
end
G = G_;

%Mesaure 2 - Sum around the well in layers
if n_index_around_well > 0
for layer = 1:n_index_around_well
    
    for well = 1:n_wells_per_dim^2
        [X_around, Y_around] = meshgrid([X_positions(well)-layer:X_positions(well)+layer],[Y_positions(well)-layer:Y_positions(well)+layer]);
        aux_position = zeros(I,J);
        aux_position(X_around(:),Y_around(:)) = 1/ ( (layer*2+1)^2 );
        G_(well,:) = aux_position(:);    
    end    
    
    G = [ G ; G_ ];
    
end
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
    simulation = FFT_MA_3D( correlation_function1, randn(4*I,4*J));
    simulation = simulation (1:I, 1:J);
    [ m_uni(:,f) ] = mean(mu_m) + 1*sqrt(C_m(1)) * reshape(simulation,I*J,1);
end

d = G*m_uni;

for it = 1:n_it
    d_per = d_obs + sqrt(n_it) * sgm * randn(size(d_obs,1),n_e);
    Am = m_uni - mean(m_uni')';
    Ad = d - mean(d')';
    C_md = Am*Ad'/(n_e-1);
    C_dd = Ad*Ad'/(n_e-1);
    m_uni = m_uni + C_md * ( ( C_dd + sqrt(n_it) * C_d ) \ ( d_per - d )) ;
    d = double( G*m_uni );
end
 
m_mean_esmda_unimodal = reshape(mean(m_uni,2),I,J);
m_std_esmda_unimodal = reshape(std(m_uni,[],2),I,J);

f = figure;
f.Position = [500 0 2000 1000];
subplot(241)
imagesc(facies)
hold all
scatter(X_positions,Y_positions,15,'r')
title('Facies ref model')
subplot(242)
imagesc(porosity)
hold all
scatter(X_positions,Y_positions,15,'r')
caxis([0.06 0.23])
title('Porosity ref model')
subplot(243)
imagesc(m_mean_esmda_unimodal)
hold all
scatter(X_positions,Y_positions,15,'r')
caxis([0.06 0.23])
title('ESMDA Unimodal')
subplot(244)
imagesc(m_std_esmda_unimodal)
hold all
scatter(X_positions,Y_positions,15,'r')
title('ESMDA Unimodal')
caxis([0 0.02])
subplot(246)
histogram(porosity)
xlim([0.0 0.3])
subplot(247)
histogram(m_uni(:))
xlim([0.0 0.3])


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

n_e = 100;
n_it = 1;
n_rep = 300;
n_rep_converge = round(0.3*n_rep);
window_size = 4;
n_facies = 2^(3^2);

P_hor = [0.75 0.25;
         0.25 0.75];     
P_ver = [0.75 0.25;
         0.25 0.75];


initial_facies = simulate_markov_2Dchain(P_hor,P_ver,ones(I,J,2));
m_ = zeros(I,J);
for elem = 1:n_e
    [ simulation1 ] = mu_m(1) + sqrt(C_m(1)) * FFT_MA_3D( correlation_function1, randn(4*I,4*J));
    simulation1 = simulation1 (1:I, 1:J);
    [ simulation2 ] = mu_m(2) + sqrt(C_m(2)) * FFT_MA_3D( correlation_function2, randn(4*I,4*J));
    simulation2 = simulation2 (1:I, 1:J);
    m_(initial_facies == 1) = simulation1(initial_facies == 1);
    m_(initial_facies == 2) = simulation2(initial_facies == 2);
    m_gm(:,elem) = m_(:);
end

prob_post = zeros(I,J);
m_mean_esmda_gm = zeros(I,J);
facies_sample_prob = zeros(I,J);
m = zeros(I,J);
facies_sample = initial_facies;

for rep = 1:n_rep 
rep / n_rep 
random_path = randperm(length(1:J*I));

for idx = random_path
       
    [i,j] = ind2sub([I,J],idx );
    
    mu_m_ = zeros(size(facies_sample));
    mu_m_(facies_sample==1) = mu_m(1);
    mu_m_(facies_sample==2) = mu_m(2);
    
    mu_m_(idx) = mu_m(1);
    mu_m_entire(:,1) = mu_m_(:);
    mu_m_(idx) = mu_m(2);
    mu_m_entire(:,2) = mu_m_(:);        
    
    % define the ensenble by varying  only the values at the sorted location idx
    for f = 1:n_e
        k(f) = (rand <= lambda(2) ) + 1;
        m_gm(idx,f) = mu_m(:,k(f)) + sqrt(C_m(k(f))) * randn;
    end
    
    m1 = m_gm(:,k==1);
    m2 = m_gm(:,k==2);
    
    %for it = 1:n_it
        
        
        n_e1 = size(m1,2);
        n_e2 = size(m2,2);
        
        d1 = double( G*m1 );
        d_per1 = d_obs + sqrt(n_it) * sgm * randn(size(d_obs,1),n_e1);
        
        Am1 = m1 - mean(m1')';
        Ad1 = d1 - mean(d1')';
        C_md1 = Am1*Ad1'/(n_e1 -1);
        C_dd1 = Ad1*Ad1'/(n_e1 -1);
                
        d2 = double( G*m2 );
        d_per2 = d_obs + sqrt(n_it) * sgm * randn(size(d_obs,1),n_e2);
        
        Am2 = m2 - mean(m2')';
        Ad2 = d2 - mean(d2')';
        C_md2 = Am2*Ad2'/(n_e2 -1);
        C_dd2 = Ad2*Ad2'/(n_e2 -1);
        
        
        facies_sample1 = facies_sample;
        facies_sample1(i,j) = 1;
        facies_sample2 = facies_sample;
        facies_sample2(i,j) = 2;
        prior_markov(1) = compute_Markov_prior(facies_sample1,P_hor,P_ver);
        prior_markov(2) = compute_Markov_prior(facies_sample2,P_hor,P_ver);         
        lambda_esmda(1) = prior_markov(1) * lambda(1) * mvnpdf( d_obs, double( G*mu_m_entire(:,1) ), diag(diag(C_dd1 + C_d)) );
        lambda_esmda(2) = prior_markov(2) * lambda(2) * mvnpdf( d_obs, double( G*mu_m_entire(:,2) ), diag(diag(C_dd2 + C_d)) );        
        %lambda_esmda(1) = lambda(1) * mvnpdf( d_obs, double( G*mu_m_entire(:,1) ), C_dd1 + C_d  );
        %lambda_esmda(2) = lambda(2) * mvnpdf( d_obs, double( G*mu_m_entire(:,2) ), C_dd2 + C_d  );
        lambda_esmda = lambda_esmda/sum(lambda_esmda);
                                
        for f = 1:n_e
            l_(f) = (rand <= lambda_esmda(2) ) + 1;
            m_gm(idx,f) = mu_m(:,l_(f)) + sqrt(C_m(l_(f))) * ( m_gm(idx,f) - mu_m(:,k(f)) ) / sqrt(C_m(k(f)));
        end                
        k = l_;    
                
        facies_sample(idx) = (rand <= lambda_esmda(2) ) + 1;
    
    
        
    %end         
    
end
    
    m_ = zeros(I,J);
    for elem = 1:n_e
        [ simulation1 ] = mu_m(1) + sqrt(C_m(1)) * FFT_MA_3D( correlation_function1, randn(4*I,4*J));
        simulation1 = simulation1 (1:I, 1:J);
        [ simulation2 ] = mu_m(2) + sqrt(C_m(2)) * FFT_MA_3D( correlation_function2, randn(4*I,4*J));
        simulation2 = simulation2 (1:I, 1:J);    
        m_(facies_sample == 1) = simulation1(facies_sample == 1);
        m_(facies_sample == 2) = simulation2(facies_sample == 2);
        m_gm(:,elem) = m_(:);
    end
    
    d = double( G*m_gm );
    d_per = d_obs + sqrt(n_it) * sgm * randn(size(d_obs,1),n_e);
    Am = m_gm - mean(m_gm')';
    Ad = d - mean(d')';
    C_md = Am*Ad'/(n_e-1);
    C_dd = Ad*Ad'/(n_e-1);
    m_gm = m_gm + C_md * ( ( C_dd + sqrt(n_it) * C_d ) \ ( d_per - d )) ;
    
    
    if rep >= n_rep_converge
        m_mean_esmda_gm = m_mean_esmda_gm + reshape(m_gm(:,1),I,J)/(n_rep-n_rep_converge);
        facies_sample_prob = facies_sample_prob + (facies_sample-1)/(n_rep-n_rep_converge);
    end

    subplot(121)
    imagesc(facies_sample)
    subplot(122)
    imagesc(facies_sample_prob)
    drawnow    
    
end

%m_mean_esmda_gm_last = reshape(mean(m,2),I,J);
%m_std_esmda_gm = reshape(std(m,[],2),I,J);

figure
imagesc(1-prob_post)

f = figure;
f.Position = [500 0 2000 1000];
subplot(241)
imagesc(facies)
hold all
scatter(X_positions,Y_positions,15,'r')
title('Facies ref model')
subplot(242)
imagesc(porosity)
hold all
scatter(X_positions,Y_positions,15,'r')
caxis([0.06 0.23])
title('Porosity ref model')
subplot(243)
imagesc(m_mean_esmda_gm)
hold all
scatter(X_positions,Y_positions,15,'r')
caxis([0.06 0.23])
title('ESMDA GM')
subplot(244)
imagesc(facies_sample_prob)
hold all
scatter(X_positions,Y_positions,15,'r')
title('ESMDA FACIES GM')
subplot(246)
histogram(porosity)
xlim([0.0 0.3])
subplot(247)
histogram(m_gm)
xlim([0.0 0.3])

figure
subplot(131)
imagesc(facies)
hold all
scatter(X_positions,Y_positions,15,'r')
title('Facies ref model')
subplot(132)
imagesc(facies_sample_prob)
hold all
scatter(X_positions,Y_positions,15,'r')
title('Facies 1 probability')
subplot(133)
imagesc(facies_sample_prob)
hold all
caxis([0.5 0.51])
scatter(X_positions,Y_positions,15,'r')
title('Most likely Facies')

% ALL POSSIBLE FACIES CONFIGS FROM CHAT GPT
% configurations = dec2bin(0:2^(window_size * window_size )-1)-'0';
% configurations = reshape(configurations,[],window_size,window_size );
% configurations(configurations==0) = 2;
% configurations = flipud(configurations );
% 
% % Defining Prior means
% mu_m_ = zeros(size(configurations));
% mu_m_(configurations==1) = mu_m(1);
% mu_m_(configurations==2) = mu_m(2);
% 
% % Defining Prior covs (assuming the same for both facies for simplicity)
% L_corr = 2;
% C_m_matrix_window = covariance_matrix_exp(C_m(1)*ones(window_size,1),L_corr,1);
% C_m_matrix_window = repmat(C_m_matrix_window,window_size,window_size);
% L = chol(C_m_matrix_window + 0.01*mean(diag(C_m_matrix_window))*eye(size(C_m_matrix_window)) )';
%     
% for f = 1:n_e
%     facies_config(1,:,:) = simulate_markov_2Dchain(P_hor,P_ver,ones(window_size,window_size,2));
%     diffe = reshape(configurations - repmat(facies_config,n_facies,1),n_facies ,9);
%     k(f) = find(sum(abs(diffe),2)==0);
%     
%     media = mu_m_(k(f),:,:);
%     %m(:,f) = media(:) + L * randn(size(media));
%     
% end
% 
% %for facies_config = 1:
% 
% 
% 
% 
% %% EVALUATION OF GM BLI WEIGHTS
% 
% P_hor = [0.75 0.25;
%          0.25 0.75];
% P_ver = P_hor;
% 
% window_size = 3;
% 
% if I == window_size 
%     
% % ALL POSSIBLE FACIES CONFIGS FROM CHAT GPT
% configurations = dec2bin(0:2^(window_size * window_size )-1)-'0';
% configurations = reshape(configurations,[],window_size,window_size );
% configurations(configurations==0) = 2;
% configurations = flipud(configurations );
% 
% means = zeros(size(configurations));
% means(configurations==1) = mu_m(1);
% means(configurations==2) = mu_m(2);
%  
% facies_prob = zeros(I,J);
% for config = 1:size(configurations,1)
%     mulpi = means(config,:,:);
%     prob(config) = compute_Markov_prior(squeeze(configurations(config,:,:)),P_hor,P_ver);
%     prob(config) = prob(config) * mvnpdf( d_obs, G*mulpi(:), G*C_m_matrix*G' + C_d );
%     facies_prob = facies_prob + prob(config) * squeeze( configurations(config,:,:) ) ;
% end
% 
% figure 
% plot(prob)
% figure
% imagesc(facies_prob)
% 
% end


