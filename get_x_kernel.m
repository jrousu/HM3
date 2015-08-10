function get_x_kernel(chunk,testset);

% obtain x-kernel into Kx_tr or Kx_ts

global Kx;
global Kx_tr;
global Kx_ts;
global X_tr;
global X_ts;
global Ktr_filename;
global Kts_filename;

global x_datasource;

if nargin < 2
    testset = 0;
end

switch x_datasource
    case 1 % feature vectors in memory
        Kx_tr = X_tr(chunk,:)*X_tr(chunk,:)';
        Kx_ts = X_tr(chunk,:)*X_ts';
		if testset
            Kx = Kx_ts;
				
        else
            Kx = Kx_tr;
        end
    case 2 % read from file
        load_x_kernel(chunk);
    % otherwise: kernel already recides in memory
end
