function [B] = Krylov_Schur(v,A,m,k)
V = Arnoldi(v,A,m);
% schur utilizza l'output di uscita di Arnoldi come restart se gli
% autovalori non sono convergenti 

lw = []; %autovalori cercati
lu = []; % autovalori unwanted
B = [];
Hsq = V'*A*V; %Matrice quadrata di Heissemberg
[y,th] = eig(Hsq); 
resid = zeros(m,1);
%verifica convergenza arnoldi

for i=1:m

    yi = y(:,i);
    xi = V*yi;
    li = th(i,i);
    resid(i) = norm(A*xi-li*xi);

end

tol = 1e-2;
if norm(resid(end)-resid(1)) < tol
    error("autovalori convergenti");
end

[Q,T] = schur(Hsq);
lt = ordeig (T);

for i = 1 : m
    if th(i,i) > 0
        lw = [lw, th(i,i)];
    end
end

lw = sort(lw,"descend");

for i = 1:m
    if ~ismember(th(i,i), lw)
        lu = [lu, th(i,i)];
    end
end

if length(lw)>k    %che succede se gli autovalori che soddisfano la richiesta sono <k?
    lw = lw(1,1:k);
end

Bw = diag(lw, 0); 
Bu = diag(lu, 0); 

Hnew = Bw;
