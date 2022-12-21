function obj = MLETest(x,v,rp)
eta   = x(1);
alpha = x(2);
lam   = x(3);
delta = x(4);


% Log Maximum Likelihood Estimate
obj = 0;
for r = 0:(size(v,2)-2)   
    t = v(:,r+2);
    p = rp(:,r+2);
    c = delta*(r+1);
    y = (t./lam).^alpha;
    A = sum(p.*(log(alpha./(eta.*t))+alpha*c*log(t/lam)));
    B = sum(p.*(log(abs(PrabhakarFunction(-y(:),eta,eta*c,c)))));
    obj = obj + (A + B);
end

obj = -obj;

% if imag(obj)>0
%     disp(x)
% end