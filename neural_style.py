
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 100
import matplotlib.pyplot as plt

from mxnet import image
style_img = image.imread('resources/style_6.jpg')
content_img = image.imread('resources/content_1.jpg')

from mxnet import nd

rgb_mean = nd.array([0.485, 0.456, 0.406])
rgb_std = nd.array([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32')/255 - rgb_mean) / rgb_std
    return img.transpose((2,0,1)).expand_dims(axis=0)

def postprocess(img):
    img = img[0].as_in_context(rgb_std.context)
    return (img.transpose((1,2,0))*rgb_std + rgb_mean).clip(0,1)

from mxnet.gluon.model_zoo import vision as models
pretrained_net = models.vgg19(pretrained=True)
# print(pretrained_net)

# style_layers = [0,5,10,19,28]
# content_layers = [21]
style_layers = [10,28]
content_layers = [0,5,19,21]
pooling_layers = [4,9,18,27,36]

from mxnet.gluon import nn

def get_net(pretrained_net, content_layers, style_layers):
    net = nn.HybridSequential()
    for i in range(max(content_layers+style_layers)+1):
        if i in pooling_layers:
            net.add(nn.MaxPool2D(pool_size=2, strides=2))
        else:
            net.add(pretrained_net.features[i])
    return net

net = get_net(pretrained_net, content_layers, style_layers)
net.hybridize()

def extract_features(x, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        x = net[i](x)
        if i in style_layers:
            styles.append(x)
        if i in content_layers:
            contents.append(x)
    return contents, styles


def content_loss(yhat, y):
    return (yhat-y).square().mean()
#     return nd.sum((yhat-y).square()) * 0.5


def gram(x):
    c = x.shape[1]
    n = int(x.size / x.shape[1])
    y = x.reshape((c, n))
    return nd.dot(y, y.T) / n


def style_loss(yhat, gram_y):
    c = yhat.shape[1]
    n = yhat.size / yhat.shape[1]
    return (gram(yhat) - gram_y).square().mean()/ 4

def tv_loss(yhat):
    return 0.5*((yhat[:,:,1:,:] - yhat[:,:,:-1,:]).abs().mean() +
                (yhat[:,:,:,1:] - yhat[:,:,:,:-1]).abs().mean())

channels = [net[l].weight.shape[0] for l in style_layers]
style_weights = [1 for l in style_layers]
# style_weights = [1e5 /n**2 for n in channels]
# content_weights = [1]
content_weights = [1 for l in content_layers]
tv_weight = 0

print(style_weights, content_weights)

def sum_loss(loss, preds, truths, weights):
    return nd.add_n(*[w*loss(yhat, y) for w, yhat, y in zip(
        weights, preds, truths)])

def get_contents(image_shape):
    content_x = preprocess(content_img, image_shape).copyto(ctx)
    content_y, _ = extract_features(content_x, content_layers, style_layers)
    return content_x, content_y

def get_styles(image_shape, color_matching=None):
    style_x = preprocess(style_img, image_shape)
    if color_matching:
        style_x = color_matching(style_x, content_x.copyto(style_x.context))
    style_x = style_x.copyto(ctx)
    content_x.copyto(ctx)
    _, style_y = extract_features(style_x, content_layers, style_layers)
    style_y = [gram(y) for y in style_y]
    return style_x, style_y

import numpy as np
from scipy.linalg import sqrtm


def covariance(x, u=0):
    x = x - u
    c = x.shape[0]
    n = int(x.size / x.shape[0])
    y = x.reshape((c, -1))
    return nd.dot(y, y.T) / n

def cholesky_color_matching(style_img, content_img):
    
    content_img = content_img[0]
    style_img = style_img[0]

    content_u = nd.mean(content_img, axis=(1,2)).reshape((-1, 1, 1))
    style_u = nd.mean(style_img, axis=(1,2)).reshape((-1, 1, 1))  
    
    content_covariance = covariance(content_img, content_u)
    style_covariance = covariance(style_img, style_u)
    
    L_c = nd.linalg_potrf(content_covariance)
    L_s = nd.linalg_potrf(style_covariance)
    L_s_inv = nd.array(np.mat(L_s.asnumpy()).I) 
    
    A = nd.dot(L_c, L_s_inv)
    b = content_u - nd.dot(A, style_u)
    
    return (nd.dot(A, style_img) + b).expand_dims(axis=0)

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


from time import time
from mxnet import autograd

def train(params, max_epochs, lr, lr_decay_epoch=200):
    tic = time()
    trainer = gluon.Trainer(params, 'sgd', {'learning_rate': lr})
    
    for i in range(max_epochs):
        x = params.get('generated_image')
        with autograd.record():
            content_py, style_py = extract_features(
                x.data(), content_layers, style_layers)
            content_L  = sum_loss(
                content_loss, content_py, content_y, content_weights)
            style_L = sum_loss(
                style_loss, style_py, style_y, style_weights)
            
#             tv_L = tv_loss(x.data())
            
            loss = content_L + 500 * style_L
            
        loss.backward()
        trainer.step(1)
        
        # add sync to avoid large mem usage
        nd.waitall()

        if i % 40 == 0:
#             print('epoch %3d, content %.3f, style %.3f, tv %.3f, time %.1f sec' % (
#                 i, content_L.asscalar(), style_L.asscalar(), tv_L.asscalar(), time()-tic))
            print('epoch %3d, content %.3f, style %.3f, time %.1f sec' % (
                i, content_L.asscalar(), style_L.asscalar(), time()-tic))
            tic = time()

        if i and i % lr_decay_epoch == 0:
            lr *= 0.5
            trainer.set_learning_rate(lr)
            print('change lr to ', lr)        
    
    return params



import sys
sys.path.append('..')
import utils
import mxnet as mx
from mxnet import gluon

image_shape = (400,300)

ctx = utils.try_gpu()
net.collect_params().reset_ctx(ctx)

content_x, content_y = get_contents(image_shape)
style_x, style_y = get_styles(image_shape)

# plt.imshow(postprocess(content_x).asnumpy())
# plt.show()
# plt.imshow(postprocess(style_x).asnumpy())
# plt.show()

# x = mx.gluon.Parameter('generated_image', shape=(1, 3, 300, 400))
# x.initialize(ctx=ctx)
# x.set_data(content_x)

# params = mx.gluon.ParameterDict()
# params.update({'generated_image':x})
# params.reset_ctx(ctx)

param_file = 'results/saved_param_transfer_6_1_color2'
params.load(param_file, ctx=ctx)

learnt_params = train(params, max_epochs=2000, lr=0.005, lr_decay_epoch=2000)
learnt_params.save('results/saved_param_transfer_6_1_color2')

y = learnt_params.get('generated_image')

plt.imshow(postprocess(y.data()).asnumpy())
plt.show()
plt.imsave('results/transfer_6_1_color2.png', postprocess(y.data()).asnumpy())