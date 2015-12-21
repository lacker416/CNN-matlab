function net = skynet_bp(net, y)
    n = numel(net.layers);
    net.e = net.o - y;
    %%  backprop deltas
     if strcmp(net.layers{n}.loss,'LLH')    
       net.L = -1/size(net.e, 2)*sum(sum(y.*log(net.o)))+net.lamda/2*sum(sum(net.layers{n}.ffw.^2));       
       net.od = net.e;
     elseif strcmp(net.layers{n}.loss,'MSD')    
       net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);
       net.od = net.e .* (net.o .* (1 - net.o));        
     end
     net.layers{n}.d{1} = (net.layers{n}.ffw' * net.od);    

    for l = (n - 1) : -1 : 1
        if strcmp(net.layers{l}.type, 'c')
             if strcmp(net.layers{l+1}.type, 'o') || strcmp(net.layers{l+1}.type, 'i')
                sa = size(net.layers{l}.a{1});
                fvnum = sa(1) * sa(2);
                for j = 1 : numel(net.layers{l}.a)
                    net.layers{l}.d{j} = reshape(net.layers{l+1}.d{1}(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
                    if strcmp(net.layers{l}.activetype,'relu')
                        net.layers{l}.d{j} = net.layers{l}.d{j} .* (net.layers{l}.a{j} > 0);
                    elseif strcmp(net.layers{l}.activetype,'sigmoid')
                        net.layers{l}.d{j} = net.layers{l}.d{j} .* net.layers{l}.a{j} .* (1 - net.layers{l}.a{j});
                    end
                end                     
             else
                 for j = 1 : net.layers{l}.outputmaps
                     if strcmp(net.layers{l}.activetype,'relu')
                         net.layers{l}.d{j} = (expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) / net.layers{l + 1}.scale ^ 2) .* (net.layers{l}.a{j} > 0);
                     else 
                         net.layers{l}.d{j} = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j}) .* (expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) / net.layers{l + 1}.scale ^ 2);
                     end
                end                              
             end
                
        elseif strcmp(net.layers{l}.type, 's')
            if strcmp(net.layers{l+1}.type, 'o') || strcmp(net.layers{l+1}.type, 'i')
                sa = size(net.layers{l}.a{1});
                fvnum = sa(1) * sa(2);
                for j = 1 : numel(net.layers{l}.a)
                    net.layers{l}.d{j} = reshape(net.layers{l+1}.d{1}(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
                end                
            else    
                 for i = 1 : net.layers{l}.outputmaps
                    z = zeros(size(net.layers{l}.a{1}));
                    if net.layers{l+1}.pad == 0
                        for j = 1 :net.layers{l+1}.outputmaps
                             z = z + convn(net.layers{l + 1}.d{j}, rot90(rot90(net.layers{l + 1}.k{i}{j})), 'full');
                        end
                    else
                       t = net.layers{l + 1}.d{j}(1:end - net.layers{l+1}.pad,1:end - net.layers{l+1}.pad,:);
                       for j = 1 :net.layers{l+1}.outputmaps
                            z = z + convn(t, rot90(rot90(net.layers{l + 1}.k{i}{j})), 'full');
                       end
                    end
                       net.layers{l}.d{i} = z;
                end                             
            end        
        elseif strcmp(net.layers{l}.type, 'i')
            if strcmp(net.layers{l}.activetype,'relu')
                net.layers{l}.d{1} = net.layers{l}.ffw' * (net.layers{l+1}.d{1} .* (net.layers{l}.a{1} > 0));
            elseif strcmp(net.layers{l}.activetype,'sigmoid')
                net.layers{l}.d{1} = net.layers{l}.ffw' * (net.layers{l+1}.d{1} .* net.layers{l}.a{1} .* (1 - net.layers{l}.a{1}));
            end
        end
    end

    %%  calc gradients
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for i = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
                end
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            end
        end
        if strcmp(net.layers{l}.type, 'i')
            net.layers{l}.dffw = net.layers{l}.d{1} * (net.layers{l}.fv)' / size(net.layers{l}.d{1}, 2);
            net.layers{l}.dffb = mean(net.layers{l}.d{1}, 2);         
        end
        if strcmp(net.layers{l}.type, 'o')
          if strcmp(net.layers{l}.loss,'softmax')     
            net.layers{l}.dffw = net.od * (net.fv)' / size(net.od, 2) + net.lamda * net.layers{l}.ffw;
          else
            net.layers{l}.dffw = net.od * (net.fv)' / size(net.od, 2);
            net.layers{l}.dffb = mean(net.od, 2);         
          end            
        end
    end
end
