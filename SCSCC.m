function [c,x] = SCSCC(y, B, K, classes)
% y : data
% B : Dictionary of (Subsampled,truncated) Signed Distance Functions
% K : sparsity
% classes : class labels for dictionary
% c : class
% x : weights
% Gunnar Atli Sigurdsson, 2014

    const = 200*norm(B(:,1))*ones(size(B,1),1);
    Phi = [B const -const]; % adding constants to dictionary
    [~,N] = size(Phi);
    x = zeros(1,N);
    S = []; % positions indexes of components of s
    res = y; % first residual
    PhiS = []; % Matrix of the columns used to represent y
    options = optimset('Algorithm', 'interior-point-convex', ...
			'Display', 'off', ...
		    'Diagnostics', 'off');
    normPhi = sqrt(sum(Phi.^2,1))';

    % project residual onto dictionary, and reconstruct under convexity constraint
    for t=1:K;
	remaining = setdiff(1:N, S);
	tmp = Phi(:, remaining)'*res;
	[~,j]=max(tmp./normPhi(remaining));
	j = remaining(j);
	S = [S j];
	PhiS = [PhiS Phi(:,j)];
	% invert with convex combination constraint
	x_est = quadprog(2*(PhiS'*PhiS), -2*y'*PhiS, [], [], ...
		ones(1,size(PhiS,2)), 1, zeros(size(PhiS,2),1), [], [], options);
	res = y - PhiS*x_est;
	x(S) = x_est;
    end
    x = x(1:end-2);

    % find the class with smallest residual
    residues = zeros(max(classes),1);
    for k=1:max(classes)
        yh = B*(x.*(classes == k));
	residues(k) = norm(yh - G);
    end
    [~, c] = min(residues);
end
