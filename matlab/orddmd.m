function [lam, w, z] = orddmd(Y0, Y1, r)
% ORDDMD Dynamic mode decomposition based on SVD.
%
% See the following papers for details.
%   Schmid: Dynamic mode decomposition of numerical and experimental data,
%   Journal of Fluid Mechanics, 656:5-28, 2010.
%   Tu et al.: On dynamic mode decomposition: theory and applications, Journal
%   of Computational Dynamics, 1(2):391-421, 2014.

if nargin<3, r=rank(Y0); end

[Ur, Sr, Vr] = reducedsvd(Y0, r);
M = Y1*Vr*diag(1./diag(Sr));
tilA = Ur'*M;
[tilw, lam, tilz] = eig(tilA);
lam = diag(lam);
w = M*tilw*diag(1./lam);
z = Ur*tilz;
for i=1:length(lam), z(:,i) = z(:,i)/(w(:,i)'*z(:,i)); end

end

% ----------

function [Ur, Sr, Vr, contratio] = reducedsvd(X, r)

if nargin<2, r=rank(X); end

[Ur, Sr, Vr] = svd(X, 'econ');
diagSr = diag(Sr);
contratio = cumsum(diagSr)/sum(diagSr);
%r = min(r, sum(diagSr>eps));
Ur = Ur(:,1:r);
Sr = Sr(1:r, 1:r);
Vr = Vr(:,1:r);

end