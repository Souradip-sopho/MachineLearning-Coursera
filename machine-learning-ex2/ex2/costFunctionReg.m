function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
thetaT_X =(theta'*X')';
g_thetaT_X=sigmoid(thetaT_X);
for i=1:m
    J = J+(-y(i)*log(g_thetaT_X(i))-(1-y(i))*log(1-g_thetaT_X(i))); 
end
J=J/m;
[r,c]=size(theta);
for j=2:r
    J=J+(lambda/(2*m))*(theta(j))^2;
end

grad = (((g_thetaT_X-y)'*X))';
grad=grad/m;
for j=2:r
    grad(j)=grad(j)+(lambda/m)*theta(j);
end


% =============================================================

end
