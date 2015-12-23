function net = skynet_setup(net, x, y)
    n = numel(net.layers);
    for l = 1 : n   %  layer
        if strcmp(net.layers{l}.type, 'd')
            if net.layers{l}.channel == 1
                net.layers{l}.mapsize = size(squeeze(x(:, :, 1)));
                net.layers{l}.outputmaps = 1;
            elseif net.layers{l}.channel == 3
                net.layers{l}.mapsize = size(squeeze(x(:,:,1,1)));
                net.layers{l}.outputmaps = 3;                              
            end
        end        
        if strcmp(net.layers{l}.type, 's')
            net.layers{l}.mapsize = net.layers{l-1}.mapsize / net.layers{l}.scale;
            net.layers{l}.outputmaps = net.layers{l-1}.outputmaps;  
        end
        if strcmp(net.layers{l}.type, 'c')
            net.layers{l}.mapsize = net.layers{l-1}.mapsize - net.layers{l}.kernelsize + 1 + net.layers{l}.pad;
            fan_out = net.layers{l}.outputmaps * net.layers{l}.kernelsize ^ 2;
            fan_in = net.layers{l-1}.outputmaps * net.layers{l}.kernelsize ^ 2; 
            for j = 1 : net.layers{l}.outputmaps  %  output map
                for i = 1 : net.layers{l-1}.outputmaps  %  input map
                    net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                end
                net.layers{l}.b{j} = 0;
            end
        end
        if strcmp(net.layers{l}.type, 'i')
            net.layers{l}.outputmaps = 1;
            fvnum = prod(net.layers{l-1}.mapsize) * net.layers{l-1}.outputmaps;
            onum = net.layers{l}.mapsize;
            net.layers{l}.ffb = zeros(onum, 1);
            net.layers{l}.ffw = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));        
        end        
        if strcmp(net.layers{l}.type, 'o')
            fvnum = prod(net.layers{l-1}.mapsize) * net.layers{l-1}.outputmaps;
            onum = size(y, 1);
            if strcmp(net.layers{l}.loss,'LLH')    
                net.layers{l}.ffw = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));
            elseif strcmp(net.layers{l}.loss,'MSD')
                net.layers{l}.ffb = zeros(onum, 1);
                net.layers{l}.ffw = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));        
            end                   
        end
    end
end
