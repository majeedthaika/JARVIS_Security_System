def test_NN(weights_file='trained_NN_weights.pickle'):
    import pickle

    caffe_root = '/home/cs00/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)

    import sys
    sys.path.insert(0, caffe_root + 'python')
    import caffe

    caffe.set_device(0)
    caffe.set_mode_gpu()

    import numpy as np
    from pylab import *
    import tempfile

    # Helper function for deprocessing preprocessed images, e.g., for display.
    def deprocess_net_image(image):
        image = image.copy()              # don't modify destructively
        image = image[::-1]               # BGR -> RGB
        image = image.transpose(1, 2, 0)  # CHW -> HWC
        image += [123, 117, 104]          # (approximately) undo mean subtraction

        # clamp values in [0, 255]
        image[image < 0], image[image > 255] = 0, 255

        # round and cast from float32 to uint8
        image = np.round(image)
        image = np.require(image, dtype=np.uint8)

        return image


    NUM_STYLE_IMAGES = sum(1 for line in open('train/labels.txt','r+'))
    NUM_TEST_IMAGES = sum(1 for line in open('test/labels.txt','r+'))
    NUM_STYLE_LABELS = sum(1 for line in open('name_id.txt','r+'))

    import os

    # os.system(caffe_root+"data/ilsvrc12/get_ilsvrc_aux.sh")
    # os.system(caffe_root+"scripts/download_model_binary.py models/bvlc_reference_caffenet")

    # weights = caffe_root+'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    # assert os.path.exists(weights)

    style_label_file = 'name_id.txt'
    style_labels = list(np.loadtxt(style_label_file, str, delimiter='\n'))
    if NUM_STYLE_LABELS > 0:
        style_labels = style_labels[:NUM_STYLE_LABELS]
    print '\nLoaded style labels:\n', ', '.join(style_labels)


    # ### 2.  Defining and running the nets
    # 
    # We'll start by defining `caffenet`, a function which initializes the *CaffeNet* architecture (a minor variant on *AlexNet*), taking arguments specifying the data and number of output classes.

    # In[6]:

    from caffe import layers as L
    from caffe import params as P

    weight_param = dict(lr_mult=1, decay_mult=1)
    bias_param   = dict(lr_mult=2, decay_mult=0)
    learned_param = [weight_param, bias_param]

    frozen_param = [dict(lr_mult=0)] * 2

    def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
                  param=learned_param,
                  weight_filler=dict(type='gaussian', std=0.01),
                  bias_filler=dict(type='constant', value=0.1)):
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                             num_output=nout, pad=pad, group=group,
                             param=param, weight_filler=weight_filler,
                             bias_filler=bias_filler)
        return conv, L.ReLU(conv, in_place=True)

    def fc_relu(bottom, nout, param=learned_param,
                weight_filler=dict(type='gaussian', std=0.005),
                bias_filler=dict(type='constant', value=0.1)):
        fc = L.InnerProduct(bottom, num_output=nout, param=param,
                            weight_filler=weight_filler,
                            bias_filler=bias_filler)
        return fc, L.ReLU(fc, in_place=True)

    def max_pool(bottom, ks, stride=1):
        return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

    def caffenet(data, label=None, train=True, num_classes=1000,
                 classifier_name='fc8', learn_all=False):
        """Returns a NetSpec specifying CaffeNet, following the original proto text
           specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
        n = caffe.NetSpec()
        n.data = data
        param = learned_param if learn_all else frozen_param
        n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=param)
        n.pool1 = max_pool(n.relu1, 3, stride=2)
        n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
        n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
        n.pool2 = max_pool(n.relu2, 3, stride=2)
        n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
        n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=param)
        n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)
        n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
        n.pool5 = max_pool(n.relu5, 3, stride=2)
        n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)
        if train:
            n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
        else:
            fc7input = n.relu6
        n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=param)
        if train:
            n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
        else:
            fc8input = n.relu7
        # always learn fc8 (param=learned_param)
        fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
        # give fc8 the name specified by argument `classifier_name`
        n.__setattr__(classifier_name, fc8)
        if not train:
            n.probs = L.Softmax(fc8)
        if label is not None:
            n.label = label
            n.loss = L.SoftmaxWithLoss(fc8, n.label)
            n.acc = L.Accuracy(fc8, n.label)
        # write the net to a temporary file and return its filename
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(str(n.to_proto()))
            return f.name


    # Define a function `style_net` which calls `caffenet` on data from the Flickr style dataset.
    # 
    # The new network will also have the *CaffeNet* architecture, with differences in the input and output:
    # 
    # - the input is the Flickr style data we downloaded, provided by an `ImageData` layer
    # - the output is a distribution over 20 classes rather than the original 1000 ImageNet classes
    # - the classification layer is renamed from `fc8` to `fc8_flickr` to tell Caffe not to load the original classifier (`fc8`) weights from the ImageNet-pretrained model
    # In[8]:

    def style_net(train=True, learn_all=False, subset=None):
        if subset is None:
            subset = 'train' if train else 'test'
        source = '%s/labels.txt' % subset
        transform_param = dict(mirror=train, crop_size=227,
            mean_file=caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto')
        if (subset == 'train'):
            style_data, style_label = L.ImageData(
                transform_param=transform_param, source=source,
                batch_size=NUM_STYLE_IMAGES, new_height=256, new_width=256, ntop=2)
        else:
            style_data, style_label = L.ImageData(
                transform_param=transform_param, source=source,
                batch_size=NUM_TEST_IMAGES, new_height=256, new_width=256, ntop=2)
        return caffenet(data=style_data, label=style_label, train=train,
                        num_classes=NUM_STYLE_LABELS,
                        classifier_name='fc8_muic17',
                        learn_all=learn_all)


    def disp_preds(net, image, labels, k=5, name='ImageNet'):
        input_blob = net.blobs['data']
        net.blobs['data'].data[0, ...] = image
        probs = net.forward(start='conv1')['probs'][0]
        top_k = (-probs).argsort()[:k]
        print 'top %d predicted %s labels =' % (k, name)
        print '\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p])
                        for i, p in enumerate(top_k))

    def disp_style_preds(net, image):
        disp_preds(net, image, style_labels, name='style')


    def top_prediction(net, image, labels, k=1, name='ImageNet'):
        input_blob = net.blobs['data']
        net.blobs['data'].data[0, ...] = image
        probs = net.forward(start='conv1')['probs'][0]
        top_k = (-probs).argsort()[:k]
        return labels[top_k]

    def top_prediction_style(net, image):
        return top_prediction(net, image, style_labels, name='style')


    # ### 3. Training the style classifier
    # 
    # Now, we'll define a function `solver` to create our Caffe solvers, which are used to train the network (learn its weights).  In this function we'll set values for various parameters used for learning, display, and "snapshotting" -- see the inline comments for explanations of what they mean.  You may want to play with some of the learning parameters to see if you can improve on the results here!

    from caffe.proto import caffe_pb2

    def solver(train_net_path, test_net_path=None, base_lr=0.001):
        s = caffe_pb2.SolverParameter()

        # Specify locations of the train and (maybe) test networks.
        s.train_net = train_net_path
        if test_net_path is not None:
            s.test_net.append(test_net_path)
            s.test_interval = 1000  # Test after every 1000 training iterations.
            s.test_iter.append(100) # Test on 100 batches each time we test.

        # The number of iterations over which to average the gradient.
        # Effectively boosts the training batch size by the given factor, without
        # affecting memory utilization.
        s.iter_size = 1
        
        s.max_iter = 100000     # # of times to update the net (training iterations)
        
        # Solve using the stochastic gradient descent (SGD) algorithm.
        # Other choices include 'Adam' and 'RMSProp'.
        s.type = 'SGD'

        # Set the initial learning rate for SGD.
        s.base_lr = base_lr

        # Set `lr_policy` to define how the learning rate changes during training.
        # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
        # every `stepsize` iterations.
        s.lr_policy = 'step'
        s.gamma = 0.1
        s.stepsize = 20000

        # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
        # weighted average of the current gradient and previous gradients to make
        # learning more stable. L2 weight decay regularizes learning, to help prevent
        # the model from overfitting.
        s.momentum = 0.9
        s.weight_decay = 5e-4

        # Display the current training loss and accuracy every 1000 iterations.
        s.display = 1000

        # Snapshots are files used to store networks we've trained.  Here, we'll
        # snapshot every 10K iterations -- ten times during training.
        s.snapshot = 10000
        s.snapshot_prefix = caffe_root + 'models/finetune_flickr_style/finetune_flickr_style'
        
        # Train on the GPU.  Using the CPU to train large networks is very slow.
        s.solver_mode = caffe_pb2.SolverParameter.GPU
        
        # Write the solver to a temporary file and return its filename.
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(str(s))
            return f.name


    # Now we'll invoke the solver to train the style net's classification layer.
    # 
    # For the record, if you want to train the network using only the command line tool, this is the command:
    # 
    # <code>
    # build/tools/caffe train \
    #     -solver models/finetune_flickr_style/solver.prototxt \
    #     -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
    #     -gpu 0
    # </code>
    # 
    # However, we will train using Python in this example.
    # 
    # We'll first define `run_solvers`, a function that takes a list of solvers and steps each one in a round robin manner, recording the accuracy and loss values each iteration.  At the end, the learned weights are saved to a file.
    # In[17]:

    def run_solvers(niter, solvers, disp_interval=10):
        """Run solvers for niter iterations,
           returning the loss and accuracy recorded each iteration.
           `solvers` is a list of (name, solver) tuples."""
        blobs = ('loss', 'acc')
        loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                     for _ in blobs)
        for it in range(niter):
            for name, s in solvers:
                s.step(1)  # run a single SGD step in Caffe
                loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                                 for b in blobs)
            if it % disp_interval == 0 or it + 1 == niter:
                loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                      (n, loss[n][it], np.round(100*acc[n][it]))
                                      for n, _ in solvers)
                print '%3d) %s' % (it, loss_disp)     
        # Save the learned weights from both nets.
        weight_dir = tempfile.mkdtemp()
        weights = {}
        for name, s in solvers:
            filename = 'weights.%s.caffemodel' % name
            weights[name] = os.path.join(weight_dir, filename)
            s.net.save(weights[name])
        return loss, acc, weights

    def most_common(lst):
        return max(set(lst), key=lst.count)

    with open(weights_file, 'rb') as f:
        weights = pickle.load(f)
        face_recog_net = caffe.Net(style_net(train=False), weights, caffe.TEST)
        
        predictions = []
        for batch_index in range(NUM_TEST_IMAGES):
            image = face_recog_net.blobs['data'].data[batch_index]
            predictions.append(top_prediction_style(face_recog_net, image))

        return most_common(predictions)

