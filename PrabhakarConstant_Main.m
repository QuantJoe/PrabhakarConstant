clear
% clc

% read the data
M = readmatrix('C:\Users\lostm\Documents\SP_112222_noExt.csv');
M(:,1:2) = [];
M(1561:end,:) = []; %clean up data, there's a footer in this dataset


%tick over tick closing price change
s = (diff(M(:,4)));

%iterate through the ticks
k = 0; sj = []; scum = 0;
k3 = 0; k2 = 0; B = sign(s(1));
for j = 1:length(s)

    if sign(s(j))~=B && s(j)~=0 %if the sign B changes, and the change in tick isn't 0 continue

%         if B == 1 %Commented out a condition wherein only when the tick
%         changes from negative to positive does it count.
            k2 = k2 + 1;
            sj(k2,2) = k; %sum of log V between flips
            sj(k2,1) = k3;%total sum of log V for all flips
            sj(k2,3) = abs(scum); %total value increased
%         end

        k3 = k3 + (k);

        k = 0; B = -B; scum = 0; %flip B
    end
    scum = scum + s(j); %collect number of 
    k = k + log(M(j,5)); %log(Volume)
end

sj(:,1:2) = (sj(:,1:2))/100; %normalize this for numerical improvements

r = randi(length(sj)-10,1000,1); %choose 1000 random instances of 10
for j = 1:10
    r = [r,r(:,1)+j];
end


rp = arrayfun(@(x)sj(x,3),r); %pull samples of price moves

r = arrayfun(@(x)sj(x,1),r); %pull samples of sum log(V)
r = r-r(:,1); %set initial values to 0 at every select sequence
r(:,1) = [];  %eliminate the initial value 0
v(:,1) = (1:length(r))/length(r); %initialize empiracal probabilities
% close all

opts.Colors     = get(groot,'defaultAxesColorOrder');
opts.width      = 8;
opts.height     = 6;
opts.fontType   = 'Times';
opts.fontSize   = 9;

fig = figure; clf
hold on
for j = 1:10
    v(:,j+1) = sort(r(:,j)); %sort all 1000 instances of sequence step to create distributions
    plot(v(:,j+1)*100,v(:,1),'r.') %plot empirical distributions
end


% Setup optimization
options = optimoptions('fmincon','algorithm','interior-point','UseParallel',true,'ScaleProblem', 'obj-and-constr','display','none','TolFun',1e-6,'TolX',1e-6,'MaxFunEvals',5500,'MaxIter',500);%,'FiniteDifferenceType','central','DerivativeCheck','on');

% Declare Log Maximum Liklihood Estimater, and set eta = alpha = 1
obj = @(x)MLETest([1,    1,    x],v,rp);

% set bounds for (1/lambda) and delta
%[e,a,1/l,d];
lb = [0.01,0.01];%[0.5000,0.10,0.0001,0.01];
ub = [3.00,6.00];%[0.9999,3.00,3.0000,3.0];
x0 = [0.10,2.00];%[0.7500,0.75,1.0000,1.0];


out = fmincon(@(x)obj(x),x0,[],[],[],[],lb,ub,[],options);


out = [1,   1,    out(1),out(2)];

disp([1,   1,    1/out(3),out(4)])

% 
delta  = out(4);
lam    = out(3); %this is 1 / lambda
eta    = out(1);
alpha  = out(2);
% 
x      = eps:0.001:max(max(v(:,2:end)));
y      = (x./lam).^alpha;

% figure(1)
hold on
t = 0;
for r = 0:30 %plot first 30 states, as black line
    c      = delta*(r+1);
    E_H = PrabhakarFunction(-y(:),eta,eta*c+1,c);

    t = ((x(:)/lam).^(abs(c)*alpha)).*E_H;
    plot(x*100,t,'k') %plot fit distribution
end
axis tight
xlabel('Aggregate log(V_i)')
ylabel('P_r')

fig.Units               = 'centimeters';
fig.Position(3)         = 8;
fig.Position(4)         = 6;
set(fig.Children, ...
    'FontName',     'Times', ...
    'FontSize',     9);
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02))
fig.PaperPositionMode   = 'auto';