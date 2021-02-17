import numpy as np
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
# from sklearn.preprocessing import OneHotEncoder

def read_idx(filename, flatten=True):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        if len(shape)==3 and flatten:
            shape = (shape[0], shape[1]*shape[2])
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def mixture(n = 10000, pi = 0.5, path=None, flatten = True):
    if path is None:
        path = '/Users/smaity/projects/mnist-label-shift/MNIST'
    # Download data from here http://yann.lecun.com/exdb/mnist/
    x_train = read_idx(path + '/train-images-idx3-ubyte', flatten)
    y_train = read_idx(path + '/train-labels-idx1-ubyte', flatten)
    y_train = (np.copy(y_train) > 4).astype('float32')
    x_0, x_1 = x_train[y_train == 0], x_train[y_train == 1]
    n1, n0 = int(n * pi), n - int(n * pi)
    idx_0 = np.random.choice(x_0.shape[0], size=n0, replace=True)
    idx_1 = np.random.choice(x_1.shape[0], size=n1, replace=True)
    x_resampled_1, x_resampled_0 = x_1[idx_1], x_0[idx_0]
    y_resampled = np.array([0]*n0 +[1]*n1).astype('float32')
    x_resampled = np.concatenate((x_resampled_0, x_resampled_1), axis = 0)
    idx = np.random.choice(n, size=n, replace=False)
    return x_resampled[idx].astype('float32'), y_resampled[idx]



    
def read_lecun_mnist(path=None, flatten=True):
    if path is None:
        path = '/Users/smaity/projects/mnist-label-shift/MNIST'
    # Download data from here http://yann.lecun.com/exdb/mnist/
    X_train = read_idx(path + '/train-images-idx3-ubyte', flatten)
    y_train = read_idx(path + '/train-labels-idx1-ubyte', flatten)
    X_test = read_idx(path + '/t10k-images-idx3-ubyte', flatten)
    y_test = read_idx(path + '/t10k-labels-idx1-ubyte', flatten)
    
    scale_const = 255
    X_train = X_train.astype('float32')/scale_const
    X_test = X_test.astype('float32')/scale_const
    
    
    idx_train = np.random.choice(X_train.shape[0], size=12000, replace=False)
    return X_train[idx_train], y_train[idx_train], X_test, y_test

def binarize(y, label_noise=0.25):
    
    y = np.copy(y) > 4
    y = np.logical_xor(y, np.random.binomial(1, label_noise, size=len(y)))
    
    return y.astype(int)
    
def color_digits(X, y, color_noise, downsample=True):
    
    if downsample:
        X = np.copy(X)[:,::2,::2]
    
    color = np.logical_xor(y, np.random.binomial(1, color_noise, size=len(y)))
    colored_X = np.repeat(X[:,None,:,:],2,axis=1)
    colored_X[color,0,:,:] = 0
    colored_X[~color,1,:,:] = 0
    
    colored_X = colored_X.reshape(X.shape[0],-1)
    
    return colored_X
    
    
def make_environments(data=None, downsample=True, path=None, red_0_corrs = None):
    if data is None:
        X_train, y_train, X_test, y_test = read_lecun_mnist(path=path, flatten=False)
    else:
        X_train, y_train, X_test, y_test = data
        
    y_train = binarize(y_train)
    y_test = binarize(y_test)
    
    if red_0_corrs == None: 
        red_0_corrs = [0.8, 0.9]
    
    n_envs = len(red_0_corrs)
    idx_order = list(range(X_train.shape[0]))
    np.random.shuffle(idx_order)
    idx_envs = np.array_split(idx_order, n_envs)
    
    envs = []
    for i in range(n_envs):
        env_X = color_digits(X_train[idx_envs[i]], y_train[idx_envs[i]], 1-red_0_corrs[i])
        envs.append([env_X, y_train[idx_envs[i]]])
    
    test_corr = 0.1
    
    X_test = color_digits(X_test, y_test, 1-test_corr)
    
    return envs, [X_test, y_test]
    
def plot_images(X, n_row=10, n_col=10, shape=(2,14,14), scale=False):
    fig = plt.figure(figsize=(n_row, n_col))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(n_row, n_col),  # creates 2x2 grid of axes
                     axes_pad=0.05,  # pad between axes in inch.
                     )
    for ax in grid:
        index = np.random.choice(np.arange(X.shape[0]))
        img = X[index].reshape(shape)
        if scale:
            img = img - img.min()
            img /= img.max()
        # img = (np.vstack((img, np.zeros((1,shape[1],shape[2]))))*255).astype(int)
        img = np.vstack((img, np.zeros((1,shape[1],shape[2]))))
        img = np.moveaxis(img, 0, -1)
        ax.imshow(img)
        ax.set_axis_off()
    plt.show()
    return

def grey_hist(W):
    W_as_img = W.T.reshape((-1,2,14,14))
    W_diff = np.abs(W_as_img[:,0,:,:] - W_as_img[:,1,:,:]).mean(axis=(1,2))
    _ = plt.hist(W_diff)
    plt.show()
    return

def logit_to_prob(x):
    prob = 1/(1 + np.exp(-x))
    return prob

def grey_inv(X, tf_logit, tf_X, plot=False):
    N = X.shape[0]
    X_as_img = X.reshape((-1,2,14,14))
    orig_logit = tf_logit.eval(feed_dict={tf_X: X}).flatten()
    orig_logit = logit_to_prob(orig_logit)
    coeff = [0., 0.2, 0.5, 0.7, 1.]
    total_X = X_as_img.sum(axis=1)
    inv_logits = []
    for c in coeff:
        c_X = np.zeros(X_as_img.shape)
        c_X[:,0,:,:] = total_X*c
        c_X[:,1,:,:] = total_X*(1-c)
        c_X = c_X.reshape((N,-1))
        # np.random.seed(1)
        # plot_images(c_X)
        c_logit = tf_logit.eval(feed_dict={tf_X: c_X}).flatten()
        inv_logits.append(logit_to_prob(c_logit))
    logits_std = np.std(inv_logits, axis=0)
    
    if plot:
        n_half = N // 2
        _ = plt.hist(orig_logit[:n_half])
        _ = plt.hist(orig_logit[n_half:])
        plt.show()
        _ = plt.hist(logits_std)
        plt.show()

    return orig_logit.std(), logits_std.mean()
    
