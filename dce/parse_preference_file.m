% =================== begin matlab code ====================
function file_struct=parse_preference_file(filename,verbose,keywords,defaults)
% parse_file: parses a preferences file
% usage: file_struct=parse_preference_file(filename,keywords)
%
% arguments (input)
% filename - character string containing the name of a
% preferences file. If this file is not on
% the path, then the name must also contain
% the full path to that file.
% keywords - cell array containing a list of legal
% keywords. keywords are not case sensitive,
% and only enough characters need be supplied
% make the choice of keyword un-ambiguous.
% defaults - (optional) cell array containing a list of
% default values, one for each keyword supplied
% if no defaults array is supplied, then no
% defaults will be filled in. defaults must be
% exactly the same size as keywords or defaults
% must be empty or not even supplied
%
% arguments (output)
% file_struct - structure containing one field for each
% keyword found in the keyword list
%
% parse_file ignores all empty lines, and all lines which
% start with a '%' character. These are comments.
% white space at the beginning of a line is also ignored.

delimiter = '=';
comment = '%';

% start by checking that the file exists at all
ex=exist(filename,'file');
if ex~=2
    error(['File not found: ',filename])
end

% check for defaults
file_struct=[];
if nargin<4
    defaults=[];
elseif ~isempty(defaults)
    % stuff defaults
    file_struct=[];
    for i=1:length(defaults)
        file_struct=setfield(file_struct,keywords{i},defaults{i});
    end
end

% open file
fid=fopen(filename,'r');

% get one record at a time
flag=1;
while flag
    rec=fgetl(fid);
    if ~ischar(rec)
        % then it must have been and end-of-file(-1)
        flag=0;
    else
        if ~isempty(rec)
            % ignore empty lines
            % grab the first token in the record, trim whitespace
            [tok,rest]=strtok(rec,delimiter);
            tok = strtrim(tok);
            rest = strtrim(rest);
            if isempty(tok)
                continue
            end
            % is this a comment (first character, minus whitespace, of token=='%')
            if ~strcmp(tok(1),comment)
                % check to see if this was a keyword in the list
                [key,~]=which_valid_property(tok,keywords,verbose);
                if ~isempty(key)
                    % stuff the field with the remainder of the record
                    % after dropping any leading and trailing blanks.
                    k=find(rest~=delimiter,1,'first');
                    if isempty(k)
                        rest='';
                    elseif k>1
                        rest(1:(k-1))='';
                    end
                    rest = strtrim(rest);
                    file_struct=setfield(file_struct,key,rest);
                end
            end
        end
    end
end


% when all done, be nice and close the file
fclose(fid);

% =================== begin sub-functions here ===================
% ================================================================
function [property,ind]=which_valid_property(property,valid_list,verbose)
% checks the valid list of properties supplied, and returns the
% one which matches, first looking for an exact match, then looking
% to see if the user has only supplied the first few characters of
% a valid property
%
% arguments (input):
% property - string to be checked for validity
% valid_list - cell array containing valid property strings
%
% arguments (output):
% property - string containing the full property name from valid_list
% ind - scalar index of the property identified in valid_list

ind=strmatch(lower(property),lower(valid_list),'exact');
if isempty(ind)
    ind=strmatch(lower(property),lower(valid_list));
end

% is it in the list of legal properties?
if isempty(ind)
    if verbose
        warndlg(['"',property,'" is not a valid option and will be ignored, check preference file'])
    end
    property='';
elseif length(ind)>1
    if verbose
        warndlg([property,' matches more than one possible option.', ...
            ' be more specific.'])
    end
    property='';
else
    property=valid_list{ind};
end

% =================== end of code ======================