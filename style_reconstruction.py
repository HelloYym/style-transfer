import matplotlib as mpl
mpl.rcParams['figure.dpi']= 100
import matplotlib.pyplot as plt

from mxnet import image
style_img = image.imread('resources/style_6.jpg')



from mxnet import nd

rgb_mean = nd.array([0.485, 0.456, 0.406])
rgb_std = nd.array([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    img = img.astype('float32')/255
    img = image.imresize(img, *image_shape)
    img = image.color_normalize(img, rgb_mean, rgb_std)
    return img.transpose((2,0,1)).expand_dims(axis=0)

def postprocess(img):
    img = img[0].as_in_context(rgb_std.context)
    return (img.transpose((1,2,0))*rgb_std + rgb_mean).clip(0,1)

from mxnet.gluon.model_zoo import vision as models

pretrained_net = models.vgg19(pretrained=True)


style_layers = [1,6,11,20,29]
pooling_layers = [4,9,18,27,36]

from mxnet.gluon import nn

def get_net(pretrained_net, style_layers):
    net = nn.HybridSequential()
    for i in range(max(style_layers)+1):
        if i in pooling_layers:
            net.add(nn.MaxPool2D(pool_size=2, strides=2))
        else:
            net.add(pretrained_net.features[i])
    return net

net = get_net(pretrained_net, style_layers)
net.hybridize()

def extract_features(x, style_layers):
    styles = []
    for i in range(len(net)):
        x = net[i](x)
        if i in style_layers:
            styles.append(x)
    return styles


def gram(x):
    c = x.shape[1]
    n = int(x.size / x.shape[1])
    y = x.reshape((c, n))
    return nd.dot(y, y.T)


def style_loss(yhat, gram_y):
    c = yhat.shape[1]
    n = yhat.size / yhat.shape[1]
    return (gram(yhat) - gram_y).square().sum() / (4 * yhat.size ** 2)


# channels = [net[l].weight.shape[0] for l in style_layers]
# style_weights = [1e4/n**2 for n in channels]
style_weights = [1 for l in style_layers]

def tv_loss(yhat):
    return 0.5*((yhat[:,:,1:,:] - yhat[:,:,:-1,:]).abs().mean() +
                (yhat[:,:,:,1:] - yhat[:,:,:,:-1]).abs().mean())

def sum_loss(loss, preds, truths, weights):
    return nd.add_n(*[w*loss(yhat, y) for w, yhat, y in zip(
        weights, preds, truths)])


def get_styles(image_shape):
    style_x = preprocess(style_img, image_shape).copyto(ctx)
    style_y = extract_features(style_x, style_layers)
    style_y = [gram(y) for y in style_y]
    return style_x, style_y


from time import time
from mxnet import autograd

def train(params, max_epochs, lr, lr_decay_epoch=200):
    tic = time()
    trainer = gluon.Trainer(params, 'adam', {'learning_rate': lr})
    for i in range(max_epochs):
        x = params.get('generated_image')
        with autograd.record():
            style_py = extract_features(x.data(), style_layers)
            style_L = sum_loss(
                style_loss, style_py, style_y, style_weights)
            
            tv_L = tv_loss(x.data())
            loss = style_L
            
        loss.backward()
        trainer.step(1)
        
        # add sync to avoid large mem usage
        nd.waitall()

        if i % 40 == 0:
            print('epoch %3d, style %.3f, time %.1f sec' % (
                i, loss.asscalar(), time()-tic))
            tic = time()

        if i and i % lr_decay_epoch == 0:
            lr *= 0.5
            trainer.set_learning_rate(lr)
            print('change lr to ', lr)        
    
    return params

import utils
import mxnet as mx
from mxnet import gluon

image_shape = (256,256)

ctx = mx.gpu(0)
net.collect_params().reset_ctx(ctx)

style_x, style_y = get_styles(image_shape)

# x = mx.gluon.Parameter('generated_image', shape=(1, 3, 256, 256))
# x.initialize(mx.init.Xavier(), ctx=ctx)
# params = mx.gluon.ParameterDict()
# params.update({'generated_image':x})
# params.reset_ctx(ctx)

param_file = 'results/saved_param_style_conv5_1_test'
params.load(param_file, ctx=ctx)

learnt_params = train(params, max_epochs=1000, lr=0.01, lr_decay_epoch=1000)
learnt_params.save('results/saved_param_style_conv5_1_test')

y = learnt_params.get('generated_image')

plt.imshow(postprocess(y.data()).asnumpy())
plt.show()
plt.imsave('results/style_conv5_1_test.png', postprocess(y.data()).asnumpy())