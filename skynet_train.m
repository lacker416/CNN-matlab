function net = skynet_train(net, x, y, opts)
    m = size(x, 3);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];
    
    for i = 1 : opts.numepochs
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        kk = randperm(m);
        for l = 1 : numbatches
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
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
