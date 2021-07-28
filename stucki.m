function temp = stucki(Z)
% Z is HxW matrix that goes from 0 to 1

w=size(Z,2);
h=size(Z,1);

temp=Z;
for i=1:size(Z,1)
    for j=1:size(Z,2)            
        Old_pix = temp(i,j);
        New_pix = round(Old_pix);
            
        temp(i,j) = New_pix;
            
        Error = Old_pix - New_pix;

        if (j<w-1) 
            temp(i,j+1)=temp(i,j+1) + 8/42*Error;
                
            if(i<h-1)
                temp(i+1,j+1) =temp(i+1,j+1) + 4/42*Error;
            end
                
            if(i<h-2)
                temp(i+2,j+1) = temp(i+2,j+1) + 2/42*Error;
            end
        end
                
        if (j<w-2)
            temp(i,j+2) = temp(i,j+2) + 4/42*Error;
            
            if (i<h-1)
                temp(i+1,j+2) = temp(i+1,j+2) + 2/42*Error;
            end
                
            if (i<h-2)
                temp(i+2,j+2) = temp(i+2,j+2) + 1/42*Error;
            end
        end
                
        if (i<h-1)
            temp(i+1,j) = temp(i+1,j) + 8/42*Error;
        end
                
        if (i<h-2)
            temp(i+2,j) =temp(i+2,j) + 4/42*Error;
        end
            
        
        if (j>1)
            
            if (i<h-1)
                temp(i+1,j-1) =temp(i+1,j-1) + 4/42*Error;
            end
                
            if (i<h-2)
                temp(i+2,j-1) =temp(i+2,j-1) + 2/42*Error;
            end
        end
                
        if (j>2)            
            if(i<h-1)
                temp(i+1,j-2) = temp(i+1,j-2) + 2/42*Error;
            end
                
            if(i<h-2)
                temp(i+2,j-2) =temp(i+2,j-2) + 1/42*Error;
            end
        end
    end
end
                
end

