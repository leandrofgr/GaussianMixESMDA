function [prob] = compute_Markov_prior(facies,P_hor,P_ver)

I = size(facies,1);
J = size(facies,2);

prob = 1;
for i = 1:I-1
    
    for j = 1:J-1
        
        prob = prob * P_hor(facies(i,j),facies(i+1,j)) * P_ver(facies(i,j),facies(i,j+1)) ;
        
    end
    
end