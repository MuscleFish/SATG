from tabulate import tabulate
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras import layers, callbacks, initializers, models
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

class CosSim(layers.Layer):
    def __init__(self,**kwargs):
        super(CosSim, self).__init__(**kwargs)
        
    def call(self, inputs, trainable=None):
        s = tf.reduce_sum(inputs[0] * inputs[1], axis=-1)
        s /= (tf.norm(inputs[0], axis=-1) * tf.norm(inputs[1], axis=-1) + 1e-8)
        return s
    
class L2(layers.Layer):
    def __init__(self,**kwargs):
        super(L2, self).__init__(**kwargs)
        
    def call(self, inputs, trainable=None):
        s = tf.reduce_sum((inputs[0] - inputs[1]) ** 2., axis=-1)
        s = 1. - s ** 0.5
        return s

class ThetaLayer(layers.Layer):
    def __init__(self, theta=0., is_add=False, use_bias=True, init_bias=-3., **kwargs):
        super().__init__(**kwargs)
        self.is_add = is_add
        self.theta = theta
        self.use_bias = use_bias
        self.ibias = init_bias
        
    def build(self, input_shapes):
        self.bias = self.add_weight(name='bias', shape=(1,), dtype='float32', 
                                    initializer=initializers.Constant(self.ibias))
        
    def call(self, inputs, training=None):
        act_theta = tf.sigmoid(self.theta)
        if self.use_bias:
            act_bias = tf.sigmoid(self.bias)
            act_theta += (1.-act_theta) * act_bias
        if type(inputs) != type([]):
            return inputs * act_theta
        elif len(inputs) == 1:
            return inputs[0] * act_theta
        elif self.is_add:
            return inputs[0] * act_theta + inputs[1] * (1 - act_theta)
        else:
            return [inputs[0] * act_theta, inputs[1] * (1 - act_theta)]
        
class ScoreLayer(layers.Layer):
    def __init__(self, l=2.0, **kwargs):
        super().__init__(**kwargs)
        self.l = l
        
    def build(self, input_shape):
        self.c = self.add_weight(name="center", shape=[input_shape[-1],], trainable=True)
        self.w = self.add_weight(name="weight", shape=[input_shape[-1],], trainable=True)
        self.build = True
        
    def call(self, inputs, training=None):
        act_weight = tf.tanh(self.w)
        distance = (act_weight * tf.abs(inputs - self.c)) ** self.l
        dissum = tf.reduce_sum(distance, axis=-1) ** (1 / self.l)
        return -dissum
    
class CutScoreLayer(layers.Layer):
    def __init__(self, theta_layer, scale=None, lamb=0.01, **kwargs):
        super().__init__(**kwargs)
        self.tl = theta_layer
        if scale is None:
            self._scale = self.add_weight(shape=(1,), initializer=initializers.Constant(4.), dtype='float32', name='scale')
            self.scale = layers.ReLU()(self._scale)
        else:
            self.scale = float(max(1, scale))
        self.reshape = layers.Reshape((1,))
        self.lamb = min(1., float(lamb))
        
    def call(self, inputs, training=None):
        max_score = self.reshape(tf.reduce_max(inputs, axis=-1))
        min_score = self.reshape(tf.reduce_min(inputs, axis=-1))
        mid_score = self.tl(max_score - min_score) + min_score
        # (a\times theta, a)->(-6,6)
        percent_score = (inputs - mid_score + self.lamb)/(max_score - mid_score + self.lamb * inputs.shape[-1])
        new_score = 2.*self.scale*percent_score - self.scale
#         new_score = (2.*self.scale*(inputs - mid_score)/range_score) - self.scale
#         new_score = inputs - mid_score
        sig_score = tf.sigmoid(new_score)
#         print(max_score.shape, mid_score.shape, sig_score.shape)
        return sig_score

class MultiScoreLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        ps = tf.reshape(inputs[0], shape=(-1,inputs[0].shape[-1],1))
        pc = ps*inputs[1]
        ns = 1 - ps
        nc = ns*inputs[1]
        
        return pc, nc
    
class MeanAddLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        return (inputs[0]+inputs[1])/2
    
class ChangeK(callbacks.Callback):
    def __init__(self, theta_layer, l=64, **kwargs):
        super().__init__(**kwargs)
        self.l = l
        self.i = layers.Input((1,))
        self.o = theta_layer(self.i)
        self.m = models.Model(self.i, self.o)
        self.m.summary()
        
    def on_epoch_end(self, epoch, logs=None):
        k = int(self.m.predict([[self.l]])[0,0])
        print('new K is', k)
        
class TopKLayer(layers.Layer):
    def __init__(self, theta, **kwargs):
        super().__init__(**kwargs)
        self.theta = theta
        
    def build(self, input_shape):
        self.l = input_shape[-1]
        self.build = True
        
    def call(self, inputs, training=None):
        act_theta = tf.sigmoid(self.theta)
        topk = tf.nn.top_k(inputs, k=int(self.l*(1-act_theta[0])))
        mid = tf.reduce_min(topk.values)
        print(mid)
        return tf.sigmoid(inputs - mid)

def printTable(lines):
    try:
        from IPython.display import HTML, display
        display(HTML(tabulate(lines, tablefmt='html')))
    except Exception as e:
        print(tabulate(lines))
    
def sklearn_call(y_true, y_pred):
    print(y_true.shape, y_pred.shape)
    y_pred_new = np.zeros(y_pred.shape, dtype='int32')
    for i, l in enumerate(y_pred):
        for j, v in enumerate(l):
            y_pred_new[i,j] = 1 if v >= .5 else 0
    return {
        "macro precision": precision_score(y_true, y_pred_new, average='macro'),
        "macro recall": recall_score(y_true, y_pred_new, average='macro'),
        "macro f1": f1_score(y_true, y_pred_new, average='macro'),
        "micro precision": precision_score(y_true, y_pred_new, average='micro'),
        "micro recall": recall_score(y_true, y_pred_new, average='micro'),
        "micro f1": f1_score(y_true, y_pred_new, average='micro'),
    }

class TestOnEnd(callbacks.Callback):
    def __init__(self, test_x, test_y, test_col=None, top_k:int=3, batch_size=32, **kwargs):
        super().__init__(**kwargs)
        self.test_x = test_x
        self.test_y = test_y
        self.test_col = test_col
        self.top_k = int(max(3, top_k))
        self.batch_size = batch_size
        
    def on_train_begin(self, logs=None):
        self.test_history = []
        self.has_multi=False
        
    def on_epoch_end(self, epoch=0, logs=None):
        y_pred = self.model.predict(self.test_x, batch_size=self.batch_size)
        if self.test_col is not None:
            y_pred = y_pred[self.test_col]
            self.test_history.append(sklearn_call(self.test_y, y_pred))
        elif len(y_pred[0].shape) == 2:
            history = [sklearn_call(self.test_y, pl) for pl in y_pred]
            self.test_history.append(history)
            self.has_multi = True
        else:
            self.test_history.append(sklearn_call(self.test_y, y_pred))
        print(self.test_history[-1])
        
    def on_train_end(self, logs=None):
        if not self.has_multi:
            print(sorted(self.test_history, key=lambda x:x['macro f1'], reverse=True)[:self.top_k])
        else:
            print(sorted(self.test_history, key=lambda x:x[0]['macro f1'], reverse=True)[:self.top_k])

class WarmUpLineDecayScheduler(callbacks.Callback):
    def __init__(self, lr_max,lr_min, warm_step,sum_step,bat):
        super(WarmUpLineDecayScheduler, self).__init__()
        self.lr_max      = lr_max
        self.lr_min    = lr_min
        self.warm_step    = warm_step
        self.sum_step = sum_step
        self.bat = bat

    def on_train_begin(self, batch, logs=None):
        self.init_lr = K.get_value(self.model.optimizer.lr)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        
    def on_batch_end(self,batch, logs=None):
        step = self.epoch*self.sum_step+batch
        # print('step:',step)
        learning_decay_steps = 1
        learning_decay_rate = 0.999
        warm_lr = self.lr_max * (step / self.warm_step)
        decay_lr = max(self.init_lr * tf.pow(learning_decay_rate , (step / learning_decay_steps)),self.lr_min)
        if  step < self.warm_step:
            lr = warm_lr
        else:
            lr =decay_lr
        K.set_value(self.model.optimizer.lr, lr)

class WeightAdd(layers.Layer):
    def __init__(self, **kwargs):
        super(WeightAdd, self).__init__(**kwargs)
        
    def build(self, input_shapes):
        self.w = self.add_weight(name='weight', shape=(input_shapes[-1], 1), 
                                 initializer=initializers.Ones(), dtype='float32')
        
    def call(self, x, trainable=None):
        w = self.w
        wx = tf.matmul(x, w)
        sw = tf.reduce_sum(w)
        px = wx / sw
        return px