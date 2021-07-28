if exist('dmd_handle','var') && isvalid(dmd_handle)
   % Stop all sequences
    dmd_handle.patternControl(0); 
    
    % Enter idle mode
    dmd_handle.idle;
end

