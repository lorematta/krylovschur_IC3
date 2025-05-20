function [V,it] = Krylov_Schur(v,A,m,k)

res = 1;
it = 0;
V = v;
while (res > 1e-5 && it< 500)

V = Arnoldi(V(:,1)/norm(V(:,1)),A,m);
% schur utilizza l'output di uscita di Arnoldi come restart se gli
% autovalori non sono convergenti 

Hsq = V'*A*V; %Matrice quadrata di Heissemberg
% [y,th] = eig(Hsq); 
% resid = zeros(m,1);


% %verifica convergenza arnoldi
% for i=1:m
% 
%     yi = y(:,i);
%     xi = V*yi;
%     li = th(i,i);
%     resid(i) = norm(A*xi-li*xi);
% 
% end
% 
% tol = 1e-2;
% if norm(resid(end)-resid(1)) < tol
%     error("autovalori convergenti, usare direttamente arnoldi");
% end

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
mod = false(length(lt),1);

for i=1:length(lt)
    mod(idx(i)) = true;
end

for i= 1:length(lt)
    if mod(i) && re(i)
        wanted(i) = 1;
    end
end

[Q_ord, T_ord] = ordschur(Q,T,wanted);   % Riordina la decomposizione di Schur in wanted e unwanted

Bw = T_ord(1:p,1:p);
Qw = Q_ord(:, 1:p);


 V = V * Qw; 
 [V, ~] = qr(V,0); % nuova base ortonormale (n x k)
 Hsq = V'*A*V; %Matrice quadrata di Heissemberg
[y,th] = eig(Hsq); 
resid = zeros(k,1);

%verifica convergenza
for i=1:k

    yi = y(:,i);
    xi = V*yi;
    li = th(i,i);
    resid(i) = norm(A*xi-li*xi);

end
res = max(resid);
it = it+1
end


Qw = Q_ord(:, 1:k);
V = V * Qw;
end
