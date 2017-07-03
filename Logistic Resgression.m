%% 
function theta =logisticReg()
%   梯度下降法寻找最合适的theta，使得代价函数J最小
options=optimset('GradObj','on','MaxIter',100);
inittheta=[0 0]';
theta=fminunc(@costFunc,inittheta,options);
end

%%
function [J,gradient] = costFunc(theta)
x = [0.0 0.1 0.7 1.0 1.1 1.3 1.4 1.7 2.1 2.2]';
y = [0 0 1 0 0 0 1 1 1 1]'; 
m=size(x,1);
tmp=theta(1)+theta(2)*x;        %theta'x
hypothesis=1./(1+exp(-tmp));  %logistic function
delta=log(hypothesis+0.01).*y+(1-y).*log(1-hypothesis+0.01);       %加上0.01是为了防止x为0
J=-sum(delta)/m;
gradient(1)=sum(hypothesis-y)/m;  %x0=1;
gradient(2)=sum((hypothesis-y).*x)/m;       %theta=theta-a*gradient;  gradient=-J'(theta)
end


