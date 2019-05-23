%PCA Principal Component Analysis (pca) on raw data.
%   Evaluates the Principal Component Analysis by computing
%    - The covariance matrix Sigma of X;
%    - The Eigenvectors U of Sigma;
%    - The projected data Z of the points X over Z.
%   
%   @param X  : `m` by `n` matrix of the Observation Set Variables.
%               Each row represents an observation with `n` variables.
%
%   @return U : the principal component coefficients for the data matrix X.
%               Each column contains one principal component coefficients.
%               Columns are in descending order w.r. to component variance.
%               Data are centered via SVD (singular value decomposition).
%   @return S : returns the principal component score, that represents the
%               X value projected over the U basis. The centered data could
%               be obtained by applying the multiplication S*U'.

function [U, S] = PCA(X)
    [m, n] = size(X);
    

    % ...
    % ...
    % ...
end

