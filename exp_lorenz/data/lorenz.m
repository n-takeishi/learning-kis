clear;

beta = 8/3;
sigma = 10;
rho = 28;
eps = 0.000001;
T = [0 121.7];
initV = [0 1 1.05];

objfun = @(T,X) [sigma*(X(2) - X(1)); X(1)*(rho - X(3)) - X(2); X(1)*X(2) - beta*X(3)];
options = odeset('RelTol',eps,'AbsTol',[eps eps eps/10]);
[T,X] = ode45(objfun, T, initV, options);

dlmwrite('./train.txt', X(1:10000,1));
dlmwrite('./val.txt', X(10001:15000,1));
dlmwrite('./test.txt', X(15001:25000,1));