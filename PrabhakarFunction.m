function [E,dEdL,dEdA,dEdB,dEdG] = PrabhakarFunction(z,alpha,beta,gama)

% This file has been heavily modified from its original for speed, and 
% includes optional gradient outputs.  I have a seperate version that 
% includes the hessian.  In this version the only constraint is  0<alpha<1.

% Joe Armstrong - Joe.w.armstrong2@gmail.com, below is the original notes
% for the original version of this file.


%
% Evaluation of the Mittag-Leffler (ML) function with 1, 2 or 3 parameters
% by means of the OPC algorithm [1]. The routine evaluates an approximation
% Et of the ML function E such that |E-Et|/(1+|E|) approx 1.0e-15   
%     
%
% E = ML(z,alpha) evaluates the ML function with one parameter alpha for
% the corresponding elements of z; alpha must be a real and positive
% scalar. The one parameter ML function is defined as
%
% E = sum_{k=0}^{infty} z^k/Gamma(alpha*k+1)
%
% with Gamma the Euler's gamma function.
%
%
% E = ML(z,alpha,beta) evaluates the ML function with two parameters alpha
% and beta for the corresponding elements of z; alpha must be a real and
% positive scalar and beta a real scalar. The two parameters ML function is
% defined as
%
% E = sum_{k=0}^{infty} z^k/Gamma(alpha*k+beta)
%
%
% E = ML(z,alpha,beta,gama) evaluates the ML function with three parameters
% alpha, beta and gama for the corresponding elements of z; alpha must be a
% real scalar such that 0<alpha<1, beta any real scalar and gama a real and
% positive scalar; the arguments z must satisfy |Arg(z)| > alpha*pi. The
% three parameters ML function is defined as 
%
% E = sum_{k=0}^{infty} Gamma(gama+k)*z^k/Gamma(gama)/k!/Gamma(alpha*k+beta)
%
%
% NOTE: 
% This routine implements the optimal parabolic contour (OPC) algorithm
% described in [1] and based on the inversion of the Laplace transform on a
% parabolic contour suitably choosen in one of the regions of analyticity
% of the Laplace transform.
%
%
% REFERENCES
%
%   [1] R. Garrappa, Numerical evaluation of two and three parameter
%   Mittag-Leffler functions, SIAM Journal of Numerical Analysis, 2015,
%   53(3), 1350-1369
%
%
%   Please, report any problem or comment to : 
%        roberto dot garrappa at uniba dot it
%
%   Copyright (c) 2015, Roberto Garrappa, University of Bari, Italy
%   roberto dot garrappa at uniba dot it
%   Homepage: http://www.dm.uniba.it/Members/garrappa
%   Revision: 1.4 - Date: October 8 2015
% Check inputs


beta = real(beta);
alpha = real(alpha);

if abs(alpha-beta)<eps
    alpha = beta;
end

if nargin < 4
    gama = 1 ;
    if nargin < 3
        beta = 1 ;
        if nargin < 2
            error('MATLAB:ml:NumberParameters', ...
                'The parameter ALPHA must be specified.');
        end
    end
end
if abs(gama-1) > eps
%     if alpha > 1
%         error('MATLAB:ml:ALPHAOutOfRange',...
%             ['With the three parameters Mittag-Leffler function ', ...
%             'the parameter ALPHA must satisfy 0 < ALPHA < 1']) ;
%     end
%     if min(abs(angle(z(abs(z)>eps)))) <= alpha*pi 
%         error('MATLAB:ml:ThreeParametersArgument', ...
%             ['With the three parameters Mittag-Leffler function ', ...
%             'this code works only when |Arg(z)|>alpha*pi.']);
%     end
end


% Target precision 
log_epsilon = log(10^(-15)); 
% Inversion of the LT for each element of z

[m,n,l] = size(z);

z = z(:);
E = zeros(size(z)) ;  

zv = [z(:),[1:length(z(:))]'];
zv = sortrows(zv,1);
z = zv(:,1);


if nargout>1
    [z,~,IC] = unique(z);
    idx = abs(z(:))<1e-14;%eps;
    z1(idx) = 1/gammaZ(beta);
    
    if ~isempty(z(~idx))
        y = z(~idx);
        [y1,y2,y3,y4,y5] = LTInversion(1,y,alpha,beta,gama,log_epsilon);
        z1(~idx) = y1;
        z2(~idx) = y2;
        z3(~idx) = y3;
        z4(~idx) = y4;
        z5(~idx) = y5;
    end
    
    try
        [y10,y20,y30,y40,y50] = LTInversion(1,1e-14,alpha,beta,gama,log_epsilon); %#ok<ASGLU>
        z1(idx) = y10;
        z2(idx) = y20(1)*0;
        z3(idx) = y30(1);
        z4(idx) = y40(1);
        z5(idx) = y50(1);
    catch ME
        disp(ME);
    end
    
    
    try
        E = z1(IC);
        dEdL = z2(IC);
        dEdA = z3(IC);
        dEdB = z4(IC);
        dEdG = z5(IC);
    catch ME
        disp(ME)
    end
    
    E(zv(:,2)) = E;
    dEdL(zv(:,2)) = dEdL;
    dEdA(zv(:,2)) = dEdA;
    dEdB(zv(:,2)) = dEdB;
    dEdG(zv(:,2)) = dEdG;
    
    E = reshape(E,m,n,l);
    dEdL = reshape(dEdL,m,n,l);
    dEdA = reshape(dEdA,m,n,l);
    dEdB = reshape(dEdB,m,n,l);
    dEdG = reshape(dEdG,m,n,l);
else    
    idx = abs(z(:))<1e-14;%eps;
    E(idx) = 1/gammaZ(beta);
    if ~isempty(E(~idx))
        [z1,~,IC] = unique(z(~idx));
        [z1] = LTInversion(1,z1,alpha,beta,gama,log_epsilon) ;
        E(~idx) = z1(IC);
    end
    E(zv(:,2)) = E;
    E = reshape(E,m,n,l);
end



% zv(:,1) = E;
% zv = sortrows(zv,2);
% 
% E = zv(:,1);
% E = E(:);

end
% =========================================================================
% Evaluation of the ML function by Laplace transform inversion
% =========================================================================
function [E,dEdL,dEdA,dEdB,dEdG] = LTInversion(t,lambda,alpha,beta,gama,log_epsilon)

% gama = 1;
% beta = alpha;

% Evaluation of the relevant poles
theta = angle(lambda) ;
kmin = min(ceil(-alpha/2 - theta./(2*pi)));
kmax = max(floor(alpha/2 - theta./(2*pi)));
% kmax = 2;

kidx = floor(alpha/2 - theta./(2*pi))-ceil(-alpha/2 - theta./(2*pi));

k_vett = kmin : kmax ;
s_star = exp((1/alpha).*...
    bsxfun(@plus,log(abs(lambda)),1i.*bsxfun(@plus,theta,2.*k_vett*pi)));

% if size(s_star,2) > size(s_star,1)
%     s_star = s_star';
% end


% Evaluation of phi(s_star) for each pole
phi_s_star = (real(s_star)+abs(s_star))/2 ;

% if length(k_vett)>1
%     % Sorting of the poles according to the value of phi(s_star)
%     [phi_s_star , index_s_star ] = sort(phi_s_star) ;
%     s_star = s_star(index_s_star);
%     % Deleting possible poles with phi_s_star=0
%     if size(phi_s_star,2)>0
%         for k = 1:size(phi_s_star,2)
%             if k == 1
%                 index_save = phi_s_star(:,k) > 10-15;
%             else
%                 index_save = index_save & phi_s_star(:,k) > eps;
%             end
%         end
%         s_star(~index_save, :) = [];
%         phi_s_star(~index_save, :) = [];
%     end
% end


% Inserting the origin in the set of the singularities
z0 = zeros(size(s_star,1),1);

kL = (kmax+1);
if kL > 0
    for j = 1:kL
        s_star(kidx==(j-2),1:end-(j-1)) = 0;
        phi_s_star(kidx==(j-2),1:end-(j-1)) = 0;
    end
end

s_star = [z0,s_star];
phi_s_star = [z0,phi_s_star];



J1 = size(s_star,2) ; J = J1 - 1 ;
% Strength of the singularities
p = [ max(0,-2*(alpha*gama-beta+1)),ones(1,J)*gama];
q = [ ones(1,J)*gama , +Inf];
phi_s_star = [phi_s_star, +Inf*(z0+1)] ;
% Looking for the admissible regions with respect to round-off errors

admissible_regions = ( ...
    (phi_s_star(:,1:end-1) < (log_epsilon - log(eps))/t) & ...
    (phi_s_star(:,1:end-1) < phi_s_star(:,2:end))) ;

N = size(admissible_regions,2);
M = size(admissible_regions,1);
idxN = bsxfun(@times,ones(M,N),1:N);
JJ1 = max(idxN,[],2);

% Initializing vectors for optimal parameters

max_JJ1 = max(JJ1(:));
z0 = Inf*ones(size(JJ1,1),max_JJ1);
z0(~admissible_regions) = 0;

idx = (1:size(z0,1))';
idx = repmat(idx,1,size(z0,2));
idx(:,2:end) = bsxfun(@plus,idx(:,2:end),idx(end,1:end-1).*(1:(N-1)));

idxM = idx.*admissible_regions;
idx0 = idxM==0;
idxZ = admissible_regions.*repmat(JJ1,1,N);

if sum(idx0(:))>0
    idxM(idx0) = [];
    idxN(idx0) = [];
    idxZ(idx0) = [];
end

idxM = idxM(:);
idxN = idxN(:);
idxZ = idxZ(:);

bool = idxN<idxZ;

% size(idxZ)


mu_vett(length(bool(:))) = 0;
h_vett(length(bool(:))) = 0;
N_vett(length(bool(:))) = 0;




find_region = 0;
bool1 = find(bool);
bool2 = find(~bool);
z0 = Inf*(1+z0);
% [n2,m2] = size(z0);
k = 0;
bool_v = z0;
bool_v(idxM) = bool;

while ~find_region
    k = k + 1;
    
    if ~isempty(bool1)
        [mu_vett(bool1),h_vett(bool1),N_vett(bool1)] = OptimalParam_RB ...
                        (t,phi_s_star(idxM(bool1)),phi_s_star(idxM(bool1)+M),...
                        p(idxN(bool1)),q(idxN(bool1)),log_epsilon);
    end
    
    if ~isempty(bool2)
        [mu_vett(bool2),h_vett(bool2),N_vett(bool2)] = OptimalParam_RU...
                    (t,phi_s_star(idxM(bool2)),p(idxN(bool2)),log_epsilon);
    end
    
    N_vett0 = z0;
    N_vett0(idxM) = N_vett;
    
    [N, ~] = min(N_vett0,[],2);
    if max(N) > 100%(max(N) > 100 && log_epsilon<-28)
        bool1 = find(bool_v(idxM) & N_vett0(idxM) > 100);
        bool2 = find(~bool_v(idxM) & N_vett0(idxM) > 100);
        
        log_epsilon = log_epsilon+log(10);
%     elseif max(N) > 1000 && log_epsilon>=-28
%         bool1 = find(bool_v(idxM) & N_vett0(idxM) > 1000);
%         bool2 = find(~bool_v(idxM) & N_vett0(idxM) > 1000);
%         
%         log_epsilon = log_epsilon+log(10);
% %     elseif ~(max(N) > 100 && log_epsilon<-28) && max(N)>1000
% %         
% %         bool1 = find(bool_v(idxM) & N_vett0(idxM) > 1000);
% %         bool2 = find(~bool_v(idxM) & N_vett0(idxM) > 1000);
% %         
% %         log_epsilon = log_epsilon+log(10);
        
    else
        find_region = 1 ;
    end
end

z0 = Inf*(1+z0);

mu_vett0 = z0;
mu_vett0(idxM) = mu_vett;
mu_vett = mu_vett0;

h_vett0 = z0;
h_vett0(idxM) = h_vett;
h_vett = h_vett0;

% z0 = Inf*(1+z0);
N_vett0 = z0;
N_vett0(idxM) = N_vett;
N_vett = real(N_vett0);

mu_vett(isinf(N_vett)) = +Inf;
h_vett(isinf(N_vett)) = +Inf;
N_vett(isinf(N_vett)) = +Inf;
    
% Selection of the admissible region for integration which involves the
% minimum number of nodes 
[N, iN] = min(N_vett,[],2); 
idx = (1:size(N_vett,1))'+(iN-1)*size(N_vett,1);
mu = mu_vett(idx) ; h = h_vett(idx);

[Cu,IA,IC] = unique([N,mu,h],'rows','stable');

% Evaluation of the inverse Laplace transform




% uH = SpecialUnique(h);
% nmax = max(N);
% uMu = SpecialUnique(mu);
% Nuniq = SpecialUnique(N);

bool = nargout>1;

S0(length(lambda)) = 0;
if bool
    dS0dL(length(lambda)) = 0;
    dS0dA(length(lambda)) = 0;
    dS0dB(length(lambda)) = 0;
    dS0dG(length(lambda)) = 0;
end

a1 = (alpha*gama-beta);
a2 =  alpha;

for j = 1:size(Cu,1)
    Lz = -Cu(j,1) : Cu(j,1);
    u = Cu(j,3)*Lz ;
    z = Cu(j,2)*(1i*u+1).^2 ;
    zd = -2*Cu(j,2)*u + 2*Cu(j,2)*1i;
    Lz = log(z);
    
    idxC = IC==IA(j);
    LC = lambda(idxC);
    
    idx = abs(log(LC)) - abs(alpha.*Lz)<log(eps*10);
    
    if bool
        Fden = ((z.^a2) - LC);
        LFden = log(Fden);
    end
    
    if bool
        if a1 == 0
            F = (zd./(Fden.^gama));
        else
            F = (((z.^a1).*zd)./(Fden.^gama));
        end
        LFden(idx) = a2.*Lz(idx);
        Fden(idx) = exp(LFden(idx).*gama);        
        F(idx) = (z(idx).^(-beta)).*zd(idx);
    else
        if a1 == 0
            F = (zd./(((z.^a2) - LC).^gama));
        else
            F = (((z.^a1).*zd)./(((z.^a2) - LC).^gama));
        end
%         Fden(idx) = exp(a2.*Lz(idx).*gama);        
        F(idx) = (z(idx).^(-beta)).*zd(idx);
    end

    if bool
        zF = exp(z*t).*F;
        zFxF = (zF./Fden);
        LzF = sum(zF.*Lz,2);

        S0(idxC)    = sum(zF,2);
        dS0dL(idxC) = gama*sum(zFxF,2);
        dS0dA(idxC) = -gama.*LC.*sum(zFxF.*Lz,2);
        dS0dB(idxC) = -LzF;
        dS0dG(idxC) = alpha*LzF-sum(zF.*LFden,2);
    else
        S0(idxC) = sum(exp(z*t).*F,2);
    end
end



%     if nargout>1
%         parfor j = 1:length(lambda)
%             k = -N(j) : N(j) ;
%             u = h(j)*k ;
%             z = mu(j)*(1i*u+1).^2 ;
%             zd = -2*mu(j)*u + 2*mu(j)*1i ;
%             zexp = exp(z*t) ;
% 
%             a1 = (alpha*gama-beta);
%             a2 =  alpha;
%             
%             
%             idx = abs(log(lambda(j))) - abs(alpha.*log(z))<log(eps*10);
%             
%             Fden = (z.^a2 - lambda(j));      
%             LFden = log(Fden);      
%             F = ((z.^a1)./((Fden).^gama)).*zd;
%             
%             LFden(idx) = a2.*log(z(idx));
%             Fden(idx) = exp(LFden(idx).*gama);           
%             F(idx) = (z(idx).^(-beta)).*zd(idx);
%             
%             S0(j) = sum(zexp.*F);
%             
%             dFdL = F.*(gama./Fden);
%             dS0dL(j) = sum(zexp.*dFdL);
%             
%             dFdA = -F.*(gama*lambda(j).*log(z))./Fden;
%             dS0dA(j) = sum(zexp.*dFdA);
% 
%             dFdB = -F.*log(z);
%             dS0dB(j) = sum(zexp.*dFdB);
% 
%             dFdG = F.*(alpha.*log(z)-LFden);
%             dS0dG(j) = sum(zexp.*dFdG);
%         end
%     else
%         parfor j = 1:length(lambda)
%             k = -N(j) : N(j) ;
%             u = h(j)*k ;
%             z = mu(j)*(1i*u+1).^2 ;
%             zd = -2*mu(j)*u + 2*mu(j)*1i ;
%             zexp = exp(z*t) ;
%             
%             a1 = (alpha*gama-beta);
%             a2 =  alpha;
% 
%             idx = abs(log(lambda(j))) - abs(a2.*log(z))<log(eps*10);
%             
%             F = ((z.^a1)./((z.^a2 - lambda(j)).^gama)).*zd;
%             F(idx) = (z(idx).^(-beta)).*zd(idx);
%             
% 
%             S0(j) = sum(zexp.*F);
%         end
% 
%     end
    
%     for j = 1:length(lambda)
%         
%         k = -N(j) : N(j) ;
%         u = h(j)*k ;
%         z = mu(j)*(1i*u+1).^2 ;
%         idx = abs(log(lambda(j))) - abs(alpha.*log(z))<log(eps*10);
%         if sum(idx)>0
%             load('logML.mat','logML')
%             L = length(logML.beta);
%             logML.lambda(L+1) = lambda(j);
%             logML.alpha(L+1) = alpha;
%             logML.beta(L+1) = beta;
%             logML.gama(L+1) = gama;
%             logML.z{L+1} = z(idx);
%             save('logML.mat','logML')
% %             disp(lambda(j))
% %             disp(alpha)
% %             disp(beta)
% %             disp(gama)
% %             disp(z(idx))
%         end
%     end
    
    
    
    
    
    S0 = S0(:);
    if nargout>1
        dS0dL = dS0dL(:);
        dS0dA = dS0dA(:);
        dS0dB = dS0dB(:);
        dS0dG = dS0dG(:);
    end



    try
        Integral = (h.*S0)./(2*pi*1i);
        if nargout>1
            dIntegraldL = (h.*dS0dL)./(2*pi*1i);
            dIntegraldA = (h.*dS0dA)./(2*pi*1i);
            dIntegraldB = (h.*dS0dB)./(2*pi*1i);
            dIntegraldG = (h.*dS0dG)./(2*pi*1i);
        end
    catch ME
        disp(ME)
    end


    ralpha = 1/alpha;
    m0 = size(s_star,2);
    n0 = size(s_star,1);


    Residues(length(N),1) = 0;
    dResiduesdL(length(N),1) = 0;
    dResiduesdA(length(N),1) = 0;
    dResiduesdB(length(N),1) = 0;
    dResiduesdG(length(N),1) = 0;

    idx0 = (m0-(iN+1))>=0;
%     if ~isempty(find(~idx0,1))
%     %     s_star_end = s_star(:,end);
%     %     ss_star = s_star_end(~idx0);
%     %     ss_star_beta = log(ss_star)*(1-beta) + t*ss_star;
%         Residues(~idx0) = 0;%(ralpha*exp(ss_star_beta)) ;
%         dResiduesdL(~idx0) = 0;
%         dResiduesdA(~idx0) = 0;
%     end

    idx = find(idx0);
    if nargout>1
        parfor j0 = 1:length(idx)
%             j = idx(j0);
%             idx0 = j+((iN(j)+1:m0)-1)*n0;
%             ss_star = s_star(idx0);
%             
%             log_ss_star = ralpha*(log(abs(lambda(j)))+1i*(angle(lambda(j))+k*k_vett(idx0)));
% 
%             ss_star_beta = log_ss_star*(1-beta) + t*ss_star;
%             exp_ss_star_beta = exp(ss_star_beta);
%             Residues(j0) = sum(ralpha*exp_ss_star_beta);
%             
%             dResiduesdL(j0) = ralpha*(ralpha/lambda(j)).*(ss_star-(1-beta)).*exp_ss_star_beta;
%             dResiduesdA(j0) = (ralpha^2)*(log_ss_star.*(ss_star-(1-beta))-1).*exp_ss_star_beta;
%             dResiduesdB(j0) = -ralpha*log_ss_star.*exp_ss_star_beta;
%             dResiduesdG(j0) = 0;
            
            j = idx(j0);
            idx0 = j+((iN(j)+1:m0)-1)*n0;
            ss_star = s_star(idx0);

            ss_star_beta = log(ss_star)*(1-beta) + t*ss_star;
            Residues(j0) = sum(ralpha*exp(ss_star_beta)) ;

            d_ss_star = ss_star./(alpha.*lambda(j));
            a = sum(ralpha*(ss_star.^-beta).*exp(t*ss_star).*d_ss_star.*(t*ss_star-beta+1));
            dResiduesdL(j0) = a;


            f = log(ss_star)*alpha;

            dss_star_betadL = (ss_star.^(1-beta)).*(-exp(t*ss_star)).*log(ss_star);
            dResiduesdB(j0) = sum(ralpha*dss_star_betadL) ;

            dResiduesdA(j0) = -sum(exp(ss_star_beta).*...
                ((alpha.^2).*log(ss_star)+f.*t.*ss_star-alpha.*f+alpha+f)./...
                (alpha.^3)+ralpha.*dss_star_betadL);

            dResiduesdG(j0) = 0;
        end
        Residues(idx) = Residues(1:length(idx));
        dResiduesdL(idx) = dResiduesdL(1:length(idx));
        dResiduesdB(idx) = dResiduesdB(1:length(idx));
        dResiduesdA(idx) = dResiduesdA(1:length(idx));
        dResiduesdG(idx) = dResiduesdG(1:length(idx));
    else
        parfor j0 = 1:length(idx)
            j = idx(j0);
            idx0 = j+((iN(j)+1:m0)-1)*n0;
            ss_star = s_star(idx0);

            ss_star_beta = log(ss_star)*(1-beta) + t*ss_star;
            Residues(j0) = sum(ralpha*exp(ss_star_beta));
        end
        Residues(idx) = Residues(1:length(idx));
    end

    E = Integral + Residues ;
    if nargout>1
        dEdL = dIntegraldL + dResiduesdL ;
        dEdA = dIntegraldA + dResiduesdA ;
        dEdB = dIntegraldB + dResiduesdB ;
        dEdG = dIntegraldG + dResiduesdG ;
        if isreal(lambda(1)) 
            dEdL = real(dEdL);
            dEdA = real(dEdA);
            dEdB = real(dEdB);
            dEdG = real(dEdG);
        end
    end

    if isreal(lambda(1)) 
        E = real(E) ;
    end

end
% =========================================================================
% Finding optimal parameters in a right-bounded region
% =========================================================================
function [muj,hj,Nj] = OptimalParam_RB ...
    (t, phi_s_star_j, phi_s_star_j1, pj, qj, log_epsilon)
% Definition of some constants

phi_s_star_j = phi_s_star_j(:);
phi_s_star_j1 = phi_s_star_j1(:);
pj = pj(:);
qj = qj(:);


log_eps = -36.043653389117154 ; % log(eps)
fac = 1.01 ;
conservative_error_analysis = 0 ;
% Maximum value of fbar as the ration between tolerance and round-off unit
f_max = exp(log_epsilon - log_eps) ;
% Evaluation of the starting values for sq_phi_star_j and sq_phi_star_j1
sq_phi_star_j = sqrt(phi_s_star_j) ;
threshold = 2*sqrt((log_epsilon - log_eps)/t) ;
sq_phi_star_j1 = min(sqrt(phi_s_star_j1), threshold - sq_phi_star_j) ;
% Zero or negative values of pj and qj
idx1 = find(pj < exp(log_epsilon)*10 & qj < exp(log_epsilon)*10);
idx2 = find(pj < exp(log_epsilon)*10 & qj >= exp(log_epsilon)*10);
idx3 = find(pj >= exp(log_epsilon)*10 & qj < exp(log_epsilon)*10);
idx4 = find(pj >= exp(log_epsilon)*10 & qj >= exp(log_epsilon)*10);

adm_region(length(pj(:)),1) = 0;
sq_phibar_star_j  = adm_region;
sq_phibar_star_j1 = adm_region;
f_bar = adm_region+1;%???
muj = adm_region;
hj = adm_region;
Nj = adm_region+Inf;

%First Condition
if ~isempty(idx1)
    
    sq_phibar_star_j(idx1) = sq_phi_star_j(idx1);
    sq_phibar_star_j1(idx1) = sq_phi_star_j1(idx1);
    adm_region(idx1) = 0;
end

%Second Condition
if ~isempty(idx2)
    
    sq_phibar_star_j(idx2) = sq_phi_star_j(idx2);
    f_min = fac*(sq_phi_star_j(idx2)./(sq_phi_star_j1(idx2)-sq_phi_star_j(idx2))).^qj(idx2);
    f_min(sq_phi_star_j(idx2) <= 0) = fac;
    idx0 = f_min<f_max;    
    
    f_bar(idx2(idx0)) = f_min(idx0) + f_min(idx0)./f_max.*(f_max-f_min(idx0));
    fq = f_bar(idx2(idx0)).^(-1./qj(idx2(idx0)));
    
    sq_phibar_star_j1(idx2(idx0)) = (2*sq_phi_star_j1(idx2(idx0))-fq.*sq_phi_star_j(idx2(idx0)))./(2+fq) ;
    adm_region(idx2(idx0)) = 1;
    adm_region(idx2(~idx0)) = 0;
end

%Third Condition
if ~isempty(idx3)
    
    sq_phibar_star_j(idx3) = sq_phi_star_j(idx3);
    f_min = fac*(sq_phi_star_j1(idx3)/(sq_phi_star_j1(idx3)-sq_phi_star_j(idx3))).^pj(idx3);
    idx0 = f_min<f_max;

    f_bar(idx3(idx0)) = f_min(idx0) + f_min(idx0)./f_max.*(f_max-f_min(idx0));
    fp = f_bar(idx3(idx0)).^(-1./pj(idx3(idx0)));
    sq_phibar_star_j(idx3(idx0)) = (2*sq_phi_star_j(idx3(idx0))+fp.*sq_phi_star_j1(idx3(idx0)))./(2-fp) ;
    adm_region(idx3(idx0)) = 1;
    adm_region(idx3(~idx0)) = 0;
end

%Fourth Condition
if ~isempty(idx4)
    
    f_min = fac*(sq_phi_star_j(idx4)+sq_phi_star_j1(idx4))./...
            (sq_phi_star_j1(idx4)-sq_phi_star_j(idx4)).^max(pj(idx4),qj(idx4)) ;
    idx0 = f_min<f_max;
    if ~isempty(idx4(idx0))
        f_min = max(f_min(idx0),1.5);
        f_bar(idx4(idx0)) = f_min + f_min./f_max.*(f_max-f_min);
        fp = f_bar(idx4(idx0)).^(-1./pj(idx4(idx0)));
        fq = f_bar(idx4(idx0)).^(-1./qj(idx4(idx0)));
        if ~conservative_error_analysis
            w = -phi_s_star_j1(idx4(idx0)).*t./log_epsilon ;
        else
            w = -2*phi_s_star_j1(idx4(idx0)).*t./(log_epsilon-phi_s_star_j1(idx4(idx0)).*t) ;
        end
        den = 2+w - (1+w).*fp + fq ;
        sq_phibar_star_j(idx4(idx0)) = ((2+w+fq).*sq_phi_star_j(idx4(idx0)) + fp.*sq_phi_star_j1(idx4(idx0)))./den ;
        sq_phibar_star_j1(idx4(idx0)) = (-(1+w).*fq.*sq_phi_star_j(idx4(idx0)) ...
            + (2+w-(1+w).*fp).*sq_phi_star_j1(idx4(idx0)))./den ;
        adm_region(idx4(idx0)) = 1;
        adm_region(idx4(~idx0)) = 0;
    end
end


% Final Review
idx0 = adm_region==1;
log_epsilon0 = log_epsilon  - log(f_bar(idx0));
if ~conservative_error_analysis
    w = -sq_phibar_star_j1(idx0).^2*t./log_epsilon0 ;
else
    w = -2.*sq_phibar_star_j1(idx0).^2*t./(log_epsilon0-sq_phibar_star_j1(idx0).^2.*t) ;
end

muj(idx0) = (((1+w).*sq_phibar_star_j(idx0) + sq_phibar_star_j1(idx0))./(2+w)).^2 ;
hj(idx0) = -2*pi./log_epsilon0.*(sq_phibar_star_j1(idx0)-sq_phibar_star_j(idx0))...
    ./((1+w).*sq_phibar_star_j(idx0) + sq_phibar_star_j1(idx0)) ;
Nj(idx0) = ceil(sqrt(1-log_epsilon0./t./muj(idx0))./hj(idx0)) ;

end
% =========================================================================
% Finding optimal parameters in a right-unbounded region
% =========================================================================
function [muj,hj,Nj] = OptimalParam_RU (t, phi_s_star_j, pj, log_epsilon)
% Evaluation of the starting values for sq_phi_star_j

phi_s_star_j = phi_s_star_j(:);
pj = pj(:);

sq_phi_s_star_j = sqrt(phi_s_star_j) ;

idx0 = phi_s_star_j>0;
phibar_star_j(idx0) = phi_s_star_j(idx0)*1.01;
phibar_star_j(~idx0) = 0.01;
sq_phibar_star_j = sqrt(phibar_star_j) ;
% Definition of some constants
f_min = 1 ; f_max = 10 ; f_tar = 5 ;


phi_t = phibar_star_j*t ; 
log_eps_phi_t = log_epsilon./phi_t ;
Nj(:,1) = ceil(phi_t./pi.*(1 - 3.*log_eps_phi_t./2 + sqrt(1-2.*log_eps_phi_t))) ;
A(:,1) = pi.*Nj(:,1)'./phi_t ;
sq_muj(:,1) = sq_phibar_star_j(:).*abs(4-A(:,1))./abs(7-sqrt(1+12*A(:,1))) ;
fbar(:,1) = ((sq_phibar_star_j(:)-sq_phi_s_star_j(:))./sq_muj(:,1)).^(-pj(:)) ;
stop = (pj(:) < exp(log_epsilon)/10) | (f_min < fbar(:,1) & fbar(:,1) < f_max) | isinf(A) ;

idx = find(~stop);
if ~isempty(idx)    
    sq_phibar_star_j(idx) = f_tar.^(-1./pj(idx)).*sq_muj(idx,1) + sq_phi_s_star_j(idx) ;
    phibar_star_j(idx) = sq_phibar_star_j(idx).^2 ;
    while ~isempty(idx) 
        phi_t = phibar_star_j(idx)*t ; 
        log_eps_phi_t = log_epsilon./phi_t ;
        Nj(idx,1) = ceil(phi_t./pi.*(1 - 3.*log_eps_phi_t./2 + sqrt(1-2.*log_eps_phi_t))) ;
        A(idx,1) = pi.*Nj(idx,1)'./phi_t ;
        sq_muj(idx,1) = rot90(sq_phibar_star_j(idx)).*abs(4-A(idx,1))./abs(7-sqrt(1+12*A(idx,1))) ;
        fbar(idx,1) = ((rot90(sq_phibar_star_j(idx))-sq_phi_s_star_j(idx))./sq_muj(idx,1)).^(-pj(idx)) ;
        stop = (pj(idx) < exp(log_epsilon)/10) | (f_min < fbar(idx,1) & fbar(idx,1) < f_max) ;
        
        idx = idx(~stop);
        if ~isempty(idx)
            sq_phibar_star_j(idx) = f_tar.^(-1./pj(idx)).*sq_muj(idx,1) + sq_phi_s_star_j(idx) ;
            phibar_star_j(idx) = sq_phibar_star_j(idx).^2 ;            
        end
    end
end


muj = sq_muj.^2 ;
hj = (-3*A - 2 + 2*sqrt(1+12*A))./(4-A)./Nj;

% Adjusting integration parameters to keep round-off errors under control
log_eps = log(eps) ; 
threshold = (log_epsilon - log_eps)/t ;
idx0 = find(muj>threshold);
idx01 = abs(pj(idx0))<exp(log_epsilon)/10;
Q( idx01) = 0;
Q(~idx01) = f_tar.^(-1./pj(idx0(~idx01))).*sqrt(muj(idx0(~idx01))) ;
% size(Q)
% size(phi_s_star_j(idx0))
if size(phi_s_star_j,1)<size(phi_s_star_j,2)
    phi_s_star_j = rot90(phi_s_star_j);
end

phibar_star_j(idx0) = bsxfun(@plus,Q',sqrt(phi_s_star_j(idx0))).^2 ;
idx02 = phibar_star_j(idx0) < threshold;
w = sqrt(log_eps/(log_eps-log_epsilon)) ;
u = sqrt(-phibar_star_j(idx0(idx02))*t/log_eps) ;
muj(idx0(idx02)) = threshold ;
Nj(idx0(idx02)) = ceil(w.*log_epsilon./2./pi./(u.*w-1)) ;
hj(idx0(idx02)) = sqrt(log_eps/(log_eps - log_epsilon))./Nj(idx0(idx02));
Nj(idx0(~idx02)) = +Inf ; 
hj(idx0(~idx02)) = 0 ;



end
