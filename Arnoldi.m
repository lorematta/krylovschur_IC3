function [V, H, r, b] = Arnoldi(v,A, m)

if norm(v) ~= 1
    error("initial vector with no unitary norm");
end

n = length(v);

if m > n
    error("Krylov subspace cannot have dimension bigger than original vector space");
end

H = [];
V = v;
r = [];
tol = 1e-12;

for i = 1:m-1
    r = A*v;
    h = V'*r;
    r = r - V*h; % se a questo punto si crea r = 0 signfica che il sottospazio di krylov ha raggiunto la sua
                 % massima dimensione 
    b = norm(r);

    if b < tol
        fprintf("breakdown numerico")
        return 
    end

    if i == 1
        H = [0,h;0,b];
    else
    H = [H, h; zeros(1,i), b];
    end
    v = r/b;
    V (:,i+1) = v;
end
r = A*v;
h = V'*r;
H = [H,h];

end
