function Yh = makehankel(Y,k)

[n,m] = size(Y);

Yh = zeros(n*k, m-k+1);
for i=1:m-k+1
    Yh(:,i) = reshape(Y(:,i:i+k-1), n*k, 1);
end

end