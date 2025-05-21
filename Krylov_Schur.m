function [V,resid] = Krylov_Schur(v,A,m,k,itz)

res = 1;
it = 1;
V = v;
resid = zeros(50,k);
while (res > 1e-5 && it<= itz)

V = Arnoldi(V(:,1)/norm(V(:,1)),A,m);
% schur utilizza l'output di uscita di Arnoldi come restart se gli
% autovalori non sono convergenti 

Hsq = V'*A*V; %Matrice quadrata di Heissemberg

[Q,T] = schur(Hsq);
lt = eig(T);
[~,idx] = sort(lt,"descend");
re = real(lt)>0;
p = sum(re);

if p == 0
    disp("Nessun autovalore wanted trovato.");
    return;
end 

wanted = false(length(lt),1);
valabs = false(length(lt),1);

for i=1:length(lt)
    valabs(idx(i)) = true;
    if valabs(i) && re(i)
        wanted(i) = 1;
    end
end

[Q_ord, T_ord] = ordschur(Q,T,wanted);   % Riordina la decomposizione di Schur in wanted e unwanted

Bw = T_ord(1:min(p,k),1:min(p,k));    
Qw = Q_ord(:, 1:min(p,k));


 V = V * Qw; 
 Hsq = V'*A*V; %Matrice quadrata di Heissemberg
[y,th] = eig(Hsq); 


%verifica convergenza

for i=1:k
    yi = y(:,i);
    xi = V*yi;
    li = th(i,i);
    resid(it,i) = norm(A*xi-li*xi);
end

res = max(max(resid));
it = it+1;
end
V = Arnoldi(V(:,1)/norm(V(:,1)),A,m);
Qw = Q_ord(:, 1:k);
V = V * Qw;
end
