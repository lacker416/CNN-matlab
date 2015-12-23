function net = skynet_fp(net, x)
    n = numel(net.layers);
    for l = 1 : n   %  for each layer
        if strcmp(net.layers{l}.type,'d')
            if net.layers{l}.channel == 3
                 net.layers{1}.a{1} = squeeze(x(:,:,1,:));
                 net.layers{1}.a{2} = squeeze(x(:,:,2,:));
                 net.layers{1}.a{3} = squeeze(x(:,:,3,:));
            else
                 net.layers{1}.a{1} = x;
            end
        end
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : net.layers{l}.outputmaps   %  for each output map
                %  create temp output map
                z = zeros([net.layers{l}.mapsize size(net.layers{1}.a{1},3)]);
                for i = 1 : net.layers{l - 1}.outputmaps   %  for each input map
                    if (net.layers{l}.pad == 0) 
                            z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
                    else
                       tt = zeros([net.layers{l}.mapsize + net.layers{l}.pad size(net.layers{1}.a{1},3)]);           
                       tt(1:net.layers{l}.mapsize(1),1:net.layers{l}.mapsize(2),:) = net.layers{l - 1}.a{i};                      
                       z = z + convn(tt, net.layers{l}.k{i}{j}, 'valid');
                    end
                end
                z = z + net.layers{l}.b{j};
                %  add bias, pass through nonlinearity
               if strcmp(net.layers{l}.activetype,'relu')
                 net.layers{l}.a{j} = relu(z);
               elseif strcmp(net.layers{l}.activetype,'sigmoid')
                 net.layers{l}.a{j} = sigm(z);                   
               end
            end
            %  set number of input maps to this layers number of outputmaps
        end
        if strcmp(net.layers{l}.type, 's')
            %  downsample
            for j = 1 : net.layers{l - 1}.outputmaps
                if strcmp(net.layers{l}.pooltype, 'mean')
                    z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');   %  mean pool
                    net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
                elseif strcmp(net.layers{l}.pooltype, 'max')
                    net.layers{l}.a{j} = zeros(net.layers{l}.mapsize(1),net.layers{l}.mapsize(2),size(net.layers{l - 1}.a{j},3));
                    for az = 1:size(net.layers{l - 1}.a{j},3)
                        for ai = 1:net.layers{l}.mapsize(1)
                            for aj = 1:net.layers{l}.mapsize(2)
                                t = net.layers{l - 1}.a{j}((ai-1)*net.layers{l}.scale + 1:ai*net.layers{l}.scale,(aj-1)*net.layers{l}.scale + 1:aj*net.layers{l}.scale,az);
                                net.layers{l}.a{j}(ai,aj,az) = max(max(t));
                            end
                        end
                    end
                end
            end
        end
        
        if strcmp(net.layers{l}.type, 'i')
            if strcmp(net.layers{l-1}.type, 'i')
                net.layers{l}.fv = net.layers{l-1}.a{1};
            else
                net.layers{l}.fv = [];
                for j = 1 : numel(net.layers{l-1}.a)
                    sa = size(net.layers{l-1}.a{j});
                    net.layers{l}.fv = [net.layers{l}.fv; reshape(net.layers{l-1}.a{j}, sa(1) * sa(2), sa(3))];
                end         
            end  
           z = net.layers{l}.ffw * net.layers{l}.fv + repmat(net.layers{l}.ffb, 1, size(net.layers{l}.fv, 2));
           if strcmp(net.layers{l}.activetype,'relu')
             net.layers{l}.a{1} = relu(z);
           elseif strcmp(net.layers{l}.activetype,'sigmoid')
             net.layers{l}.a{1} = sigm(z);                   
           end             
        end        
           
        if strcmp(net.layers{l}.type, 'o')
            if strcmp(net.layers{l-1}.type, 'i')
                net.fv = net.layers{l-1}.a{1};
            else
                net.fv = [];
                for j = 1 : numel(net.layers{l-1}.a)
                    sa = size(net.layers{l-1}.a{j});
                    net.fv = [net.fv; reshape(net.layers{l-1}.a{j}, sa(1) * sa(2), sa(3))];
                end         
            end
            if strcmp(net.layers{l}.loss,'LLH')
                net.o      = net.layers{l}.ffw * net.fv;
                M = bsxfun(@minus,net.o,max(net.o,[],1));
                h = exp(M);
                net.o = bsxfun(@rdivide,h,sum(h));
            elseif strcmp(net.layers{l}.loss,'MSD')
                net.o = sigm(net.layers{l}.ffw * net.fv + repmat(net.layers{l}.ffb, 1, size(net.fv, 2)));
            end             
        end
    end
end
