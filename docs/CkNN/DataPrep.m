function [A] = DataPrep(file,type,networkType,para,start,step,stop)
%reads file, runs required sparsening,creates Times and saves matlab 
%variabiles

D = readmatrix(strcat(file,'.csv'));
G = constructNetworkStructure(D,networkType,para);
A = double(G);
if strcmp(type,'sparse')
    A = sparse(A); 
end

Time = 10.^[start:step:stop];

if strcmp(networkType,'cknn')
    k = string(para(1));
    d = string(para(2));
    name = strcat(networkType,'_k_',k,'_d_',d);
else
    name = strcat(networkType,'_',para);
end

save(strcat(name,'.mat'),'Time','A');
end