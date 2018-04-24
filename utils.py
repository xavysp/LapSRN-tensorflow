import scipy
import numpy as np

def get_imgs_fn(file_name):
	return scipy.misc.imread(file_name, mode='RGB')

def augment_imgs_fn(x, add_noise=True):
	return x+0.1*x.std()*np.random.random(x.shape)

def normalize_imgs_fn(x):
    x = x * (2./ 255.) - 1.
    # x = x * (1./255.)
    return x

def truncate_imgs_fn(x):
	x = np.where(x > -1., x, -1.)
	x = np.where(x < 1., x, 1.)
	return x

def merge(images, size):
	h, w = images.shape[1], images.shape[2]
	img = np.zeros((h * size[0], w * size[1], 3))
	for idx, image in enumerate(images):
		i = idx % size[1]
		j = idx // size[1]
		img[j * h:j * h + h, i * w:i * w + w, :] = image
	return img