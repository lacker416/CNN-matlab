function net = skynet_grads(net, loopn)
    for l = 2 : numel(net.layers)
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : net.layers{l}.outputmaps
                for ii = 1 : net.layers{l-1}.outputmaps
                    if (loopn == 1)
                         net.layers{l}.k{ii}{j} = net.layers{l}.k{ii}{j} - net.bl * net.layers{l}.dk{ii}{j};
                    else
                         net.layers{l}.k{ii}{j} = net.layers{l}.k{ii}{j} - net.bl * (net.layers{l}.dk{ii}{j} + net.momentun * net.layers{l}.dk_old{ii}{j});
                    end
                    net.layers{l}.dk_old{ii}{j} = net.layers{l}.dk{ii}{j};
                end
                if (loopn == 1)
                    net.layers{l}.b{j} = net.layers{l}.b{j} - net.bl * net.layers{l}.db{j};
                else
                    net.layers{l}.b{j} = net.layers{l}.b{j} - net.bl * (net.layers{l}.db{j} + net.momentun * net.layers{l}.db_old{j});
                end
                net.layers{l}.db_old{j} = net.layers{l}.db{j};
            end
        end
        
        if strcmp(net.layers{l}.type, 'o')
             if (loopn ~= 1)
                net.layers{l}.ffw = net.layers{l}.ffw - net.bl * (net.layers{l}.dffw + net.momentun * net.layers{l}.dffw_old) ;
            else
                net.layers{l}.ffw = net.layers{l}.ffw - net.bl * net.layers{l}.dffw ;
            end
            net.layers{l}.dffw_old = net.layers{l}.dffw;

            if strcmp(net.layers{l}.loss,'MSD')    
                if (loopn == 1)
                   net.layers{l}.ffb = net.layers{l}.ffb - net.bl * net.layers{l}.dffb;
                else
                    net.layers{l}.ffb = net.layers{l}.ffb - net.bl * (net.layers{l}.dffb + net.momentun * net.layers{l}.dffb_old);
                end
                net.layers{l}.dffb_old = net.layers{l}.dffb;
            end            
        end
    end
end