import matplotlib as mpl
mpl.rcParams['figure.dpi']= 100
import matplotlib.pyplot as plt

from mxnet import image
content_img = image.imread('resources/content_6.jpg')

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
content_layers = [9,]
pooling_layers = [4,9,18,27,36]

from mxnet.gluon import nn

def get_net(pretrained_net, content_layers):
    net = nn.Sequential()
    for i in range(max(content_layers)+1):
        if i in pooling_layers:
            net.add(nn.MaxPool2D(pool_size=2, strides=2))
        else:
            net.add(pretrained_net.features[i])
    return net

net = get_net(pretrained_net, content_layers)


def extract_features(x, content_layers):
    for i in range(len(net)):
        x = net[i](x)
        if i in content_layers:
            return x

def content_loss(yhat, y):
#     return nd.sum((yhat-y).square()) * 0.5
    return (yhat-y).square().mean() * 0.5

def tv_loss(yhat):
    return 0.5*((yhat[:,:,1:,:] - yhat[:,:,:-1,:]).abs().mean() +
                (yhat[:,:,:,1:] - yhat[:,:,:,:-1]).abs().mean())


def get_contents(image_shape):
    content_x = preprocess(content_img, image_shape).copyto(ctx)
    content_y = extract_features(content_x, content_layers)
    return content_x, content_y


from time import time
from mxnet import autograd

def train(params, max_epochs, lr, lr_decay_epoch=200):
    tic = time()

    trainer = gluon.Trainer(params, 'adam', {'learning_rate': lr})
    
    for i in range(max_epochs):
        x = params.get('generated_image')
        with autograd.record():
            content_py = extract_features(x.data(), content_layers)            
            content_L = content_loss(content_py, content_y)
            
            tv_L = tv_loss(x.data())
            loss = content_L + tv_L
            
        loss.backward()
        trainer.step(1)
        
        # add sync to avoid large mem usage
        nd.waitall()

        if i % 40 == 0:
            print('batch %3d, content %.3f, time %.1f sec' % (
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

image_shape = (400,300)

ctx = mx.gpu(0)
net.collect_params().reset_ctx(ctx)

content_x, content_y = get_contents(image_shape)

# x = mx.gluon.Parameter('generated_image', shape=(1, 3, 300, 400))
# x.initialize(mx.init.Xavier(), ctx=ctx)
# params = mx.gluon.ParameterDict()
# params.update({'generated_image':x})
# params.reset_ctx(ctx)

param_file = 'results/saved_param_content_conv2_2'
params.load(param_file, ctx=ctx)

learnt_params = train(params, max_epochs=1000, lr=0.01, lr_decay_epoch=1000)
learnt_params.save('results/saved_param_content_conv2_2')

y = learnt_params.get('generated_image')

plt.imshow(postprocess(y.data()).asnumpy())
plt.show()
plt.imsave('results/content_conv2_2.png', postprocess(y.data()).asnumpy())