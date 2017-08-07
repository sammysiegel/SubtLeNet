import matplotlib as mpl
mpl.use('cairo')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import numpy.ma as ma
from keras.utils import plot_model
from keras.models import Model, load_model
from mpl_toolkits.axes_grid1 import make_axes_locatable


def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
        
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic


def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, cax=cax)


model = load_model('/data/t3home000/snarayan/BAdNet/pf/model3_pf100.h5')

all_weights = []
i = 0
for layer in model.layers:
    if layer.__class__.__name__ == "BatchNormalization":
        continue

    w = layer.get_weights()
    all_weights.append(w)
    if layer.__class__.__name__ == "LSTM":
        print w
        for ww in w:
            print ww.shape
    i += 1

#print all_weights

all_weights = np.array(all_weights)
np.save('all_weights.npy', all_weights)


# Visualize weights
print model.layers[0].get_weights()
#W = model.layers[0].W.get_value(borrow=True)
#W = np.squeeze(W)
#print("W shape : ", W.shape)

plt.figure(figsize=(15, 15))
plt.title('conv1 weights')
#nice_imshow(plt.gca(), make_mosaic(model.layers[0].get_weights(), 6, 6), cmap=cm.binary)


#plot_model(model, to_file='model.png', show_shapes = True)
