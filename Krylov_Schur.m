function V = Krylov_Schur(v,A,m,k)

res = 1;
it = 0;
V = v;
while (res > 1e-5 && it< 500)

V = Arnoldi(V(:,1),A,m);
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
    error("autovalori convergenti, usare direttamente arnoldi");
end

[Q,T] = schur(Hsq);
lt = eig(T);
wanted = real(lt)>0;
p = sum(wanted);

if p == 0
    disp("Nessun autovalore wanted trovato.");
    return;
end 

[Q_ord, T_ord] = ordschur(Q,T,wanted);   % Riordina la decomposizione di Schur in wanted e unwanted


if p>k
    Bw = T_ord(1:k,1:k);
    Qw = Q_ord(:, 1:k);
else
    Bw = T_ord(1:p,1:p);
    Qw = Q_ord(:, 1:p);
end


 V = V * Qw;       % nuova base ortonormale (n x k)
 Hsq = V'*A*V; %Matrice quadrata di Heissemberg
[y,th] = eig(Hsq); 
resid = zeros(k,1);


%verifica convergenza arnoldi
for i=1:k

    yi = y(:,i);
    xi = V*yi;
    li = th(i,i);
    resid(i) = norm(A*xi-li*xi);

end
res =  norm(resid(end)-resid(1));
it = it+1;
end
