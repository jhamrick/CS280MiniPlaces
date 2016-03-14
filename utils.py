import numpy as np
import matplotlib.pyplot as plt
import caffe

from caffe import layers as L
from train_places_net import MEAN, minivggnet, to_tempfile, get_split


def load_labels():
    # load labels
    labels_filename = 'development_kit/data/categories.txt'
    labels = np.loadtxt(labels_filename, str, delimiter=' ')[:, 0]
    return labels


def load_net(iters, i, labels):
    try:
        n = len(i)
    except TypeError:
        n = 1
        i = [i]
    
    with open(get_split('val'), 'r') as fh:
        images = np.array(fh.read().split("\n"))
        
    if n == 1:
        print("True category is: {}".format(labels[int(images[i[0]].split(" ")[1])]))
    
    source = to_tempfile("\n".join(images[i]))
    transform_param = dict(mirror=False, crop_size=96, mean_value=MEAN)
    places_data, places_labels = L.ImageData(
        transform_param=transform_param,
        source=source, root_folder='./images/', 
        shuffle=False, batch_size=n, ntop=2)
    net_path = minivggnet(
        data=places_data, labels=places_labels,
        train=False, cudnn=False, with_labels=False)
    
    net = caffe.Net(net_path, 'snapshot_vgg2/place_net_iter_{}.caffemodel'.format(iters), caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.array(MEAN)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    
    return net, transformer


# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)