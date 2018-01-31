import matplotlib as mpl
mpl.rcParams['figure.dpi']= 120
import matplotlib.pyplot as plt
import numpy as np
from mxnet import nd
from scipy.linalg import sqrtm

from mxnet import image
style_img = image.imread('resources/style_6.jpg')
content_img = image.imread('resources/content_1.jpg')

rgb_mean = nd.array([0.485, 0.456, 0.406])
rgb_std = nd.array([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32')/255 - rgb_mean) / rgb_std
    return img.transpose((2,0,1)).expand_dims(axis=0)

def postprocess(img):
    img = img[0].as_in_context(rgb_std.context)
    return (img.transpose((1,2,0))*rgb_std + rgb_mean).clip(0,1)

def covariance(x, u=0):
    x = x - u
    c = x.shape[0]
    n = int(x.size / x.shape[0])
    y = x.reshape((c, -1))
    return nd.dot(y, y.T) / n

def analogies_color_matching(style_img, content_img):
    
    content_img = content_img[0]
    style_img = style_img[0]

    content_u = nd.mean(content_img, axis=(1,2)).reshape((-1, 1, 1))
    style_u = nd.mean(style_img, axis=(1,2)).reshape((-1, 1, 1))  
    
    content_covariance = covariance(content_img, content_u)
    style_covariance = covariance(style_img, style_u)
    
    L_c = sqrtm(content_covariance.asnumpy())
    L_s = sqrtm(style_covariance.asnumpy())
    L_c = nd.array(L_c)
    L_s_inv = nd.array(np.mat(L_s).I) 
    
    A = nd.dot(L_c, L_s_inv)
    b = content_u - nd.dot(A, style_u)
    
    return (nd.dot(A, style_img) + b).expand_dims(axis=0)


image_shape = (400,300)

content_x = preprocess(content_img, image_shape)
style_x = preprocess(style_img, image_shape)
plt.imsave('results/style_6.png', postprocess(style_x).asnumpy())

plt.imshow(postprocess(style_x).asnumpy())
plt.show()
plt.imshow(postprocess(content_x).asnumpy())
plt.show()

style_x = analogies_color_matching(style_x,content_x)

plt.imshow(postprocess(style_x).asnumpy())
plt.show()
plt.imsave('results/content_6.png', postprocess(content_x).asnumpy())