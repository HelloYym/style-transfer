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
from mxnet.gluon import nn

pretrained_net = models.vgg19(pretrained=True)
swap_layer = 11

def get_net(pretrained_net, swap_layer):
    net = nn.HybridSequential()
    for i in range(swap_layer + 1):
        net.add(pretrained_net.features[i])
    return net

net = get_net(pretrained_net, swap_layer)
net.hybridize()

import sys
sys.path.append('..')
import utils
import mxnet as mx
from mxnet import gluon
from copy import deepcopy
import math
from time import time
from mxnet import autograd

def extract_patches(features, patch_size, stride):
    c, h, w = features.shape[1:]
    n_h = math.floor((h - patch_size)/stride + 1)
    n_w = math.floor((w - patch_size)/stride + 1)
    patches = nd.zeros(shape=(n_h * n_w, c, patch_size, patch_size), ctx=ctx)
    for i in range(n_h*n_w):
        h = math.floor(i / n_w)
        w = math.floor(i % n_w)
        patches[i] = features[0, :, (h*stride):(h*stride + patch_size), (w*stride):(w*stride + patch_size)]
    return patches

def get_weight(target_patches, patch_size, patch_stride):
#     convolution for computing cross correlation
    n_patches, c, k_h, k_w = target_patches.shape
    weight = deepcopy(target_patches)
    kernel = (k_h, k_w)
    num_filter = n_patches
    stride = (patch_stride, patch_stride)
#     normalize the patches to compute correlation
    for i in range(n_patches):
        norm = weight[i].norm()
        if norm < 1e-6:
            weight[i] = 0
        else:
            weight[i] = weight[i] * (1 / norm)
            
    return weight, kernel, stride, num_filter
    

image_shape = (400,300)
patch_size = 6
patch_stride = 6

ctx = utils.try_gpu()
net.collect_params().reset_ctx(ctx)

style_x = preprocess(style_img, image_shape).copyto(ctx)
target_features = net(style_x)
target_patches = extract_patches(target_features, patch_size, patch_stride)
cc_weight, cc_kernel, cc_stride, cc_num_filter = get_weight(target_patches, patch_size, patch_stride)


def tv_loss(yhat):
    return 0.5*((yhat[:,:,1:,:] - yhat[:,:,:-1,:]).abs().mean() +
                (yhat[:,:,:,1:] - yhat[:,:,:,:-1]).abs().mean())

def mse_loss(yhat, y):
    return (yhat-y).square().mean()

def patches_loss(content_features):
    c, h, w = content_features.shape[1:]  
    d_h, d_w = cc_stride
    cc = nd.Convolution(data=content_features, weight=cc_weight, kernel=cc_weight.shape[2:], stride=cc_stride, num_filter=cc_num_filter, no_bias=True)
    c_argmax = nd.argmax(cc, axis=1)
    loss = []
    for h in range(c_argmax.shape[1]):
        for w in range(c_argmax.shape[2]):
            ind = c_argmax[0, h, w]
            target_patch = target_patches[ind]
            input_patch = content_features[:,:,(h*d_h):(h*d_h + patch_size), (w*d_w):(w*d_w + patch_size)]
            loss.append(mse_loss(input_patch, target_patch))
    
    return sum(loss)/len(loss)

            
def train(params, max_epochs, lr, lr_decay_epoch=200):
    tic = time()
    trainer = gluon.Trainer(params, 'adam', {'learning_rate': lr})
    
    for i in range(max_epochs):        
        x = params.get('content_img')
        with autograd.record():
            
            content_features = net(x.data())
            patches_L = patches_loss(content_features)
            tv_L = tv_loss(x.data())
            loss = patches_L + tv_L

        loss.backward()
        trainer.step(1)
        
        # add sync to avoid large mem usage
        nd.waitall()

        if i % 20 == 0:
            print('epoch %3d, patches %.3f, tv_L %.3f, time %.1f sec' % (i, patches_L.asscalar(), tv_L.asscalar(), time()-tic))
#             print('epoch %3d, patches %.3f, time %.1f sec' % (i, patches_L.asscalar(), time()-tic))
            tic = time()

        if i and i % lr_decay_epoch == 0:
            lr *= 0.5
            trainer.set_learning_rate(lr)
            print('change lr to ', lr)        
    
    return params

content_x = preprocess(content_img, image_shape).copyto(ctx)

# x = mx.gluon.Parameter('content_img', shape=(1, 3, 300, 400))
# x.initialize(ctx=ctx)
# x.set_data(content_x)
# params = mx.gluon.ParameterDict()
# params.update({'content_img':x})

param_file = 'results/saved_param_patch_6'
params.load(param_file, ctx=ctx)

params.reset_ctx(ctx)

content_features = net(x.data())
# patches_L = patches_loss(content_features)
# print(patches_L)

learnt_params = train(params, max_epochs=200, lr=0.1, lr_decay_epoch=200)
learnt_params.save('results/saved_param_patch_6')

y = learnt_params.get('content_img')

plt.imshow(postprocess(y.data()).asnumpy())
plt.show()
plt.imsave('results/patch_6.png', postprocess(y.data()).asnumpy())