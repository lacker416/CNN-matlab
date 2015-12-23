function net = skynet_train(net, x, y, opts)
    
    a = size(x);
    m = a(end);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];
    
    for i = 1 : opts.numepochs
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        kk = randperm(m);
        for l = 1 : numbatches
            if (mod(l,100) == 0)
                disp(['sunloop:' num2str(l) '/' num2str(numbatches)]);
                disp(['loss' num2str(net.L)]);
            end
            if net.layers{1}.channel == 3
                batch_x = x(:, :,:, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            else
                batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            end
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));

            net = skynet_fp(net, batch_x);
            net = skynet_bp(net, batch_y);
            net = skynet_grads(net, i);
            if isempty(net.rL)
                net.rL(1) = net.L;
            end
            net.rL(end + 1) = net.L;
        end
    end
end
