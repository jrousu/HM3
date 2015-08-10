function print_message(msg,verbosity_level,filename)

global verbosity;


if verbosity >= verbosity_level
    fprintf('%s: %s\n',datestr(clock),msg);

    if nargin == 3
        fid = fopen(filename,'a');
        fprintf(fid,'%s: %s\n',datestr(clock),msg);
        fclose(fid);
    end
end


