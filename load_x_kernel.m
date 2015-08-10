function [success,failed] = load_x_kernel(chunk1,chunk2,testset)

global Kx;
global Kx_tr;
global Kx_ts;
global m;

global Ktr_filename;
global Kts_filename;

if nargin < 2
    chunk2 = 1:m;
end

if nargin < 3
    testset = 0;
end

if testset
    filename = Kts_filename;
else
    filename = Ktr_filename; 
end

k = strfind(filename, '.mat')
if isempty(k) % non mat-file
    if testset
        Kx_ts = zeros(length(chunk1),length(chunk2));
    else
        Kx_tr = zeros(length(chunk1),length(chunk2));
    end

    fid = fopen(filename,'r');

    row = 0; rows_read = 0; failed = []; success = [];
    while rows_read < length(chunk1)
        if feof(fid) error('Unexpected end of file'); end

        line = fgets(fid);
        row = row + 1;

        index = find(chunk1 == row);
        if  index
            rows_read = rows_read + 1;
            [krow,count] = sscanf(line,'%f ',[1,5885]);
            if count < length(chunk2);
                failed = [failed,index];
                continue;
            end;

            if testset
                Kx_ts(index,1:length(chunk2)) = krow(chunk2);
            else
                Kx_tr(index,1:length(chunk2)) = krow(chunk2);
            end
            success = [success,index];
        end

    end
    fclose(fid);
else
    if testset
        load(filename,Kx_ts);
        Kx_ts = Kx_ts(chunk1,chunk2);
    else
        load(filename,Kx_tr);
        Kx_tr = Kx_tr(chunk1,chunk2);
    end
end