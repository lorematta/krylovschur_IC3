function w = eigconv(mu, tau)
% the function transforms the eigenvalues from the propogator into
% eigenvalues of the Jacobian


tr = @(mu) log(mu)/tau;
w = tr(mu);

end
