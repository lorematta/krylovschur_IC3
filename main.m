clear all
clc

A = diag([10, 5, 1, -2, -10]);  % autovalori noti
v = randn(5,1);
v = v / norm(v);
m = 4;
k = 2;
Vk = Krylov_Schur(v,A,m,k);
[V, ~, ~, ~] = Arnoldi(Vk(:,1), A, m);
Hsq = V'*A*V; %Matrice quadrata di Heissemberg
th = eig(Hsq); %problema di Ritz e autovalori approssimati
