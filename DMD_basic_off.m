if exist('dmd_handle','var') && isvalid(dmd_handle)
   % Stop all sequences
    dmd_handle.patternControl(0); 
    
    % Enter sleep mode
    dmd_handle.sleep;
end

