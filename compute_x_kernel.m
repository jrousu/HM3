function K = compute_x_kernel(X,Xhat,normalize)

global tolerance;
global verbosity;
global chunksize;

if nargin < 3
    normalize = 0;
end

if normalize == 1
X = diag(sqrt(sum(X.*X,2)))*X;
Xhat = diag(sqrt(sum(Xhat.*Xhat,2)))*Xhat;
end

K = zeros(size(X,1),size(Xhat,1));

% K(:,:) = full(X*Xhat')+diag(ones(1,size(X,1)))*tolerance; % linear kernel
x_last = 0;
while x_last < size(X,1)
    x_first = x_last + 1;
    x_last = min(x_last + chunksize,size(X,1));

    K(x_first:x_last,:) = full(X(x_first:x_last,:)*Xhat') + tolerance;

    if verbosity > 3
        fprintf('%d ',x_last);
    end
end

if normalize == 2 % normalize kernel entries K(x,x') = K(x,x')/sqqrt(K(x,x)*K(x',x'))
    x_last = 0;
    while x_last < size(X,1)
        x_first = x_last + 1;
        x_last = min(x_last + chunksize,size(X,1));

        N = 1./sqrt(diag(K(x_first:x_last,x_first:x_last)));
        K(x_first:x_last,:) = N*K(x_first:x_last,:);
        K(:,x_first:x_last) = K(:,x_first:x_last);

        if verbosity > 3
            fprintf('%d ',x_last);
        end
    end
end



if verbosity > 3
        fprintf('\n',x_last);
    end
