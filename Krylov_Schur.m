function [V, resid, it] = Krylov_Schur(v,A,m,k,itmax)

res = 1;
it = 0;
V = Arnoldi(v,A,m);
resid = zeros(itmax,k);

while (res > 1e-5 && it<= itmax)
it = it+1;
V = Arnoldi(V(:,1)/norm(V(:,1)),A,m);
Hsq = V'*A*V; %Matrice quadrata di Heissemberg
[Q,T] = schur(Hsq);
lt = eig(T);

[~,idx] = sort(lt,"descend");
re = real(lt)> -1e-8;
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

% re_pos = find(real(lt) > 0);
% [~, idx] = sort(lt(re_pos), 'descend');
% selected = re_pos(idx(1:min(k, end)));  % indici wanted
% 
% wanted = false(length(lt),1);
% wanted(selected) = true;

% [Qw, Tw] = ordschur(Q,T,wanted);   % Riordina la decomposizione di Schur in wanted e unwanted



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

res = max(max(resid(it,:)));

end
end
