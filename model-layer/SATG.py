import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, optimizers, losses, callbacks, models, metrics, initializers
from BaseLayers import * 
from tcn.tcn import TCN

OUTINDEX = 1

class SeqTheta2:
    def __init__(self,
                 in_shape,
                 out_shape,
                 use_birnn=False,
                 use_conv=True,
                 f_type='cnn',
                 cnn_count=256,
                 theta=0,
                 theta_bias=-3.,
                 use_top_k=False,
                 score_l=2.,
                 cut_scale=10,
                 cut_lambda=0.01,
                 dense_dims=[256,128],
                 dense_act='tanh',
                 activation='sigmoid',
                 test_metrics=[
                     metrics.Precision(), metrics.Recall()
                 ],
                 test_datas=[],
                 with_f1=True,
                ):
        self.cnn_count = cnn_count
        self.f_type = f_type
        self.test_datas = test_datas
        self.in_shape = in_shape
        self.score_l = score_l
        self.use_birnn = use_birnn
        self.use_conv = use_conv
        
        self.input_layer = layers.Input(in_shape, name='input')
        if self.use_birnn:
            self.rnn_layer = layers.Bidirectional(
                layers.GRU(int(cnn_count//2), return_sequences=True), merge_mode='concat')
        self.conv1_layer = layers.Conv1D(cnn_count, kernel_size=1, padding='same', name='conv')
        self.score_layer = ScoreLayer(l=score_l, name='scoring')
        self.th = tf.Variable([float(theta),], name='theta_before_sig')
        self.theta_layer = ThetaLayer(theta=self.th, is_add=False, name='theta', init_bias=theta_bias)
        self.concat_layer = layers.Add(name='concat_theta')
        if not use_top_k:
            self.cut_layer = CutScoreLayer(theta_layer=self.theta_layer, scale=cut_scale, 
                                           lamb=cut_lambda, name='cut_theta')
        else:
            self.cut_layer = TopKLayer(theta=self.th, name='top_k')
#         self.chang_k = ChangeK(self.theta_layer)
        self.mul_layer = MultiScoreLayer(name='theta_mul_conv')
        self.max_layer = layers.GlobalMaxPool1D(name='max')
        self.dense_layers = []
        self.test_metrics = []
        for tm in test_metrics:
            self.test_metrics.append(tm)
        if with_f1:
            from random import random
            self.test_metrics.append(tfa.metrics.F1Score(out_shape, average='macro', name='macro_f1'))
            self.test_metrics.append(tfa.metrics.F1Score(out_shape, average='micro', name='micro_f1'))
        for i,dm in enumerate(dense_dims):
            self.dense_layers.append(layers.Dense(dm, activation=dense_act, name='dense_'+str(i)))
        self.dense_layers.append(layers.Dense(out_shape, activation=activation, name='output'))
        
        self.f_conv = layers.Conv1D(self.cnn_count, kernel_size=1, padding='same', name='f_conv')
        self.f_att = layers.Attention(name='f_att')
        self.f_dense = layers.Dense(self.cnn_count, name='f_dense')
        self.f_dense2 = layers.Dense(self.cnn_count, name='f_dense2')
        self.f_flat = layers.Flatten(name='f_flat')
        self.f_tcn = TCN(nb_filters=self.cnn_count, 
                        kernel_size=1, dilations=[1,2], 
                        activation='tanh', name='f_tcn',
                        return_sequences=True)
        self.f_rnn = layers.Bidirectional(layers.GRU(int(self.cnn_count//2), return_sequences=True), 
                                          merge_mode='concat')
        self.build_layers()
        
        self.init_default_models()
        
    def init_default_models(self):
        # 测试输出模型
        print("build text score model:")
        self.score_model = self.build_model(["s", "ps", "po", "no", "to"], model_name="score_model")
        print("build evaluate model:")
        self.eval_model = self.build_model(["to", "po", "tpo"], model_name="evaluate_model")
        if self.use_conv:
            print("build conv vectors out:")
            self.conv_vec_model = self.build_model(["c"], model_name="conv_vec")
        self.train_model = None
        self.last_train_outs = []
        
    def mlp(self, inputs):
        x = inputs
        for d in self.dense_layers:
            x = d(x)
        return x
    
    def get_score(self, x):
        s = self.score_layer(x)
        return s, self.cut_layer(s)
    
    def f(self, x):
        print('f_type:',self.f_type)
        if self.f_type in ['cnn', 'conv']:
            c = self.f_conv(x)
            return self.max_layer(c)
        elif self.f_type in ['acnn']:
            d = self.f_dense(x)
#             d2 = self.f_dense2(x)
            a = self.f_att([d,d])
            c = self.f_conv(a)
            return self.max_layer(c)
        elif self.f_type in ['att']:
            d = self.f_dense(x)
#             d2 = self.f_dense2(x)
            a = self.f_att([d,d])
#             c = self.f_conv(a)
            return self.max_layer(a)
        elif self.f_type in ['dense']:
            d = self.f_dense(x)
            return self.max_layer(d)
        elif self.f_type in ['tcn']:
            d = self.f_tcn(x)
            return self.max_layer(d)
        elif self.f_type in ['rnn']:
            d = self.f_rnn(x)
            return self.max_layer(d)
        elif self.f_type in ['rcnn']:
            d = self.f_rnn(x)
            c = self.f_conv(d)
            return self.max_layer(c)
        return self.max_layer(x)
        
    def build_layers(self):
        if self.use_birnn:
            c = self.rnn_layer(self.input_layer)
        elif self.use_conv:
            c = self.conv1_layer(self.input_layer)
        else:
            c = self.input_layer
        s, ps = self.get_score(c)
        
        if not self.use_conv:
            c = self.conv1_layer(c)
        pc, nc = self.mul_layer([ps, c])
        
        pv = self.f(pc)
        po = self.mlp(pv)
        nv = self.f(nc)
        no = self.mlp(nv)
        
        tpo, tno = self.theta_layer([po,no])
        
        tv = self.f(c)
        to = self.mlp(tv)
        
        wtpo, wto = self.theta_layer([tpo, to])
        mto = self.concat_layer([wtpo,wto])
        
        # pv和tv应该接近
        vcs = CosSim(name='tv_pv_cos_sim')([pv, tv])
#         vcs = L2(name='tv_pv_l2_sim')([pv,tv])
        cno = layers.Reshape((1,))(1.-vcs) * no * 100.
        
        self.out_layers = {
            "to":  [to,  True],
            "po":  [po,  True],
            "tpo": [tpo, True],
            "mto": [mto, True],
            "no":  [no,  False],
            "tno": [tno, False],
            "s":   [s,   False],
            "ps":  [ps,  False],
            "c":   [c,   False],
            "cno": [cno, False],
            "pv":  [pv,  False],
            "tv":  [tv,  False],
        }
        
        self.losses = {
            "bc_loss": losses.BinaryCrossentropy(),
            "mse_loss": losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
            "mae_loss": losses.MeanAbsoluteError()
        }
        
        self.adm = optimizers.Adamax()
        
    def build_model(self, out_names=['to','tpo','no'], model_name='SeqTheta'):
        outs = [self.out_layers[n][0] for n in out_names]
        outloss = [self.losses["bc_loss"] if self.out_layers[n][1] else self.losses["mae_loss"] for n in out_names]
        model = models.Model(
            self.input_layer,
            outs,
            name=model_name
        )
        model.compile(loss=outloss, optimizer=self.adm, metrics=self.test_metrics)
        print("use outs:",out_names)
        return model
        
    def build_train_model(self, x, y, z, out_names=['to','tpo','no'], epochs=20, batch_size=64, callbacks=[]):
        if 's' in out_names or 'ps' in out_names:
            raise ValueError("text score/percentage score cannot be train!")
        if " ".join(self.last_train_outs) != " ".join(out_names) or self.train_model is None:
            self.train_model = self.build_model(out_names if len(out_names) > 0 else ['to'])
        test_sets = [y if self.out_layers[n][1] else z for n in out_names]
#         print(len(test_sets), len(self.train_model.outputs))
        try:
            self.train_model.fit(x, test_sets, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        except Exception as e:
            print(e)
            i = len(callbacks)-1
            while i:
                try:
#                     print(callbacks[i].test_history)
                    print(callbacks[i].on_train_end())
                    break
                except Exception as e:
                    i -= 1
        finally:
            return callbacks
        
    def set_theta_bias(self, theta_use_bias=True):
        self.theta_layer.use_bias = theta_use_bias
        
    def train(self, x, y, z, use_model='all', epochs=20, batch_size=64, use_test_epoch=True, callbacks=None):
        if use_model in ['train', 'tpo,tno']:
            out_names = ['tpo','tno']
        elif use_model in ['concat', 'mto,tno']:
            out_names=['mto','tno']
        elif use_model in ['theta', 'po']:
            out_names=['po']
        elif use_model in ['conv', 'cnn', 'to']:
            out_names=['to']
        elif use_model in ['notheta4neg','to,tpo,no']:
            out_names=['to','tpo','no']
        elif use_model in ['all','to,tpo,tno']:
            out_names=['to','tpo','tno']
        else:
            out_names = use_model.split(",")
        if len(self.test_datas) < 2:
            self.test_datas = [x, y]
        test_col = out_names.index("tpo") if "tpo" in out_names else 0
        if callbacks is None:
            callbacks = [
                TestOnEnd(self.test_datas[0], self.test_datas[1], test_col)
            ] if use_test_epoch else []
        elif use_test_epoch:
            callbacks.append(TestOnEnd(self.test_datas[0], self.test_datas[1], test_col, batch_size=batch_size))
        return self.build_train_model(x=x, y=y, z=z, out_names=out_names, 
                                       epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        
    def print_test_results(self, dataset, sentence_index, word_tokens, out_labels=[], TESTINDEX=1):
        x = dataset[0][TESTINDEX:TESTINDEX+1]
        y = dataset[OUTINDEX][TESTINDEX:TESTINDEX+1]
        w = word_tokens[sentence_index[TESTINDEX]][0].split(" ")
        t = round(tf.sigmoid(self.th).numpy().tolist()[0], 3)
        print(w)
        text_score, percent_score, positive_out, negative_out, true_out = self.score_model.predict(x)
        text_zeros = tf.reduce_sum(x, axis=-1)
#         text_signs = tf.where(text_zeros!=0,1.,0.)
        print("标签预测值：")
        printTable([
            ["output_style"] + out_labels,
            ["positive"] + [round(o,2) for o in positive_out[0].tolist()],
            ["negative"] + [round(o,2) for o in negative_out[0].tolist()],
            ["original"] + [round(o,2) for o in true_out[0].tolist()],
            ["grdtruth"] + y[0].tolist()
        ])
        print("文本权重：")
        ps = [round(o,2) for o in percent_score[0,:len(w)].tolist()]
        printTable([
            ["words:"] + w,
            ["weights:"] + [round(o,2) for o in text_score[0,:len(w)].tolist()],
            ["percent:"] + ps,
            ["θ:"+str(t)] + ["√" if o > t else "" for o in ps]
        ])
        
    def print_predict(self, x, words, ttp_labels):
        outs = self.score_model.predict(x)
        for i, line in enumerate(words):
            print("words score:")
            printTable([
                ['Words:']+line,
                ['Score:']+[round(float(p),4) for p in outs[0][i]][:len(line)],
                ['Sigma:']+[round(float(p),4) for p in outs[1][i]][:len(line)]
            ])
            
            print("ttp predict:")
            print("f(X+):",[ttp_labels[j]+"-"+str(round(p,4)) for j, p in enumerate(outs[2][i]) if p > 0.5])
            print("f(X):",[ttp_labels[j]+"-"+str(round(p,4)) for j, p in enumerate(outs[4][i]) if p > 0.5])
            printTable([
                ['TTP Lable:']+ttp_labels,
                ['f(X+):']+[round(float(p),4) for p in outs[2][i]],
                ['f(X):']+[round(float(p),4) for p in outs[4][i]]
            ])
        
    def evaluate(self, x, y):
        self.eval_model.evaluate(x, y)
        
    def sklearn_matrix(self, test_x, test_y):
        return sklearn_call(test_y, self.eval_model.predict(test_x)[1])
    
    def save_all_out_model(self, save_file_path):
        all_model = self.build_model(list(self.out_layers.keys()))
        all_model.save(save_file_path)
        
    def load_models(self, save_file_path):
        all_model = models.load_model(save_file_path)
        self.input_layer = all_model.input
        last_k = None
        for i, k in enumerate(self.out_layers.keys()):
            if i < len(all_model.output):
                self.out_layers[k][0] = all_model.output[i]
                if i == len(all_model.output) - 1:
                    last_k = k
            else:
                self.out_layers[k][0] = self.out_layers[last_k][0]
        self.init_default_models()
#         self.theta = self.eval_model.output[-1].get_weight('theta')

class MultiCenterSeqTheta(SeqTheta2):
    def __init__(self,
                 score_center_num=3,
                 score_concat_type='max',
                 score_concat_before=True,
                 score_l=2.,
                 **kwargs
                ):
        self.score_layers = []
        for i in range(score_center_num):
            self.score_layers.append(ScoreLayer(l=score_l, name='scoring_'+str(i)))
        self.score_concat_layer = layers.Concatenate(name='score_concat')
        self.score_concat_type = score_concat_type
        self.score_concat_before = score_concat_before
#         self.weight_add_layer = layers.Dense(1, 
#                                              activation='softmax',
#                                              kernel_regularizer='l1',
#                                              use_bias=False,
#                                              name='weight_add')
        self.weight_add_layer = WeightAdd(name='weight_add')
        super().__init__(**kwargs)
        print("distance center nums:", len(self.score_layers), 'distance action:', score_concat_type)
        self.single_score_model = models.Model(self.input_layer, self.center_scores)
        
    def get_score(self, x):
        return self.get_score_before(x) if self.score_concat_before else self.get_score_after(x)
        
    def get_score_before(self, x):
        score_outs = []
        self.center_scores = []
        for sl in self.score_layers:
            cs = sl(x)
            csp = self.cut_layer(cs)
            self.center_scores.append(cs)
            csr = tf.reshape(csp, (-1, csp.shape[1], 1))
            score_outs.append(csr)
        co = self.score_concat_layer(score_outs)
        all_score = self.score_concat(co)
        return layers.Concatenate(axis=-1)(self.center_scores), all_score
        
    def score_concat(self, x):
        if self.score_concat_type in ['max']:
            return tf.reduce_max(x, axis=-1)
        elif self.score_concat_type in ['min']:
            return tf.reduce_min(x, axis=-1)
        elif self.score_concat_type in ['conv']:
            return layers.Flatten()(layers.Conv1D(1, kernel_size=1)(x))
        elif self.score_concat_type in ['add','dense']:
            return layers.Flatten()(self.weight_add_layer(x))
        else:
            return tf.reduce_mean(x, axis=-1)
        
    def get_score_after(self, x):
        self.center_scores = []
        for sl in self.score_layers:
            s = sl(x)
            s1 = tf.reshape(s, (-1, s.shape[-1], 1))
            self.center_scores.append(s1)
        cs = self.score_concat_layer(self.center_scores)
        cs_all = self.score_concat(cs)
        csp = self.cut_layer(cs_all)
        return layers.Concatenate(axis=-1)(self.center_scores), csp
    
    def print_test_results(self, dataset, sentence_index, word_tokens, out_labels=[], TESTINDEX=1):
        x = dataset[0][TESTINDEX:TESTINDEX+1]
        y = dataset[OUTINDEX][TESTINDEX:TESTINDEX+1]
        w = word_tokens[sentence_index[TESTINDEX]][0].split(" ")
        t = round(tf.sigmoid(self.th).numpy().tolist()[0], 3)
        print(w)
        text_score, percent_score, positive_out, negative_out, true_out = self.score_model.predict(x)
        text_zeros = tf.reduce_sum(x, axis=-1)
#         text_signs = tf.where(text_zeros!=0,1.,0.)
        print("标签预测值：")
        printTable([
            ["output_style"] + out_labels,
            ["positive"] + [round(o,2) for o in positive_out[0].tolist()],
            ["negative"] + [round(o,2) for o in negative_out[0].tolist()],
            ["original"] + [round(o,2) for o in true_out[0].tolist()],
            ["grdtruth"] + y[0].tolist()
        ])
        print("文本权重：")
        ps = [round(o,2) for o in percent_score[0,:len(w)].tolist()]
        if len(text_score[0,:len(w)].shape) >= 2:
            ws = [["weights-"+str(i)+":"]+[round(o,2) for o in l] for i, l in enumerate(
                np.transpose(text_score[0,:len(w)], (-1, -2)).tolist())]
        else:
            ws = [["weights:"]+[round(o,2) for o in text_score[0,:len(w)]]]
        printTable([
            ["words:"] + w]+ ws +[
            ["percent:"] + ps,
            ["θ:"+str(t)] + ["√" if o > t else "" for o in ps]
        ])

def SeqMask(in_shape,
            out_shape,
            use_conv=True,
            cnn_kernel_nums=[1],
            cnn_count=256,
            score_l=2.,
            dense_dims=[256,128],
            dense_act='tanh',
            activation='sigmoid',
            test_metrics=[
                metrics.Precision(), metrics.Recall()
            ],
            with_f1=True,):
    input_layer = layers.Input(in_shape, name='input')
    if type(cnn_kernel_nums) == "int":
        conv1_ls = [layers.Conv1D(cnn_count, kernel_size=cnn_kernel_nums, padding='same', name='conv')]
    else:
        conv1_ls = [layers.Conv1D(cnn_count, kernel_size=cn, padding='same', name='conv_'+str(i)) for i, cn in enumerate(cnn_kernel_nums)]
    conv1_layer = layers.Concatenate(axis=-1)
    score_layer = ScoreLayer(l=score_l, name='scoring')
    soft_layer = layers.Softmax(axis=-1, name='softmax')
    max_layer = layers.GlobalMaxPool1D(name='max')
    dense_layers = []
    tms = []
    for tm in test_metrics:
        tms.append(tm)
    if with_f1:
        tms.append(tfa.metrics.F1Score(out_shape, average='macro', name='macro_f1'))
        tms.append(tfa.metrics.F1Score(out_shape, average='micro', name='micro_f1'))
    for i,dm in enumerate(dense_dims):
        dense_layers.append(layers.Dense(dm, activation=dense_act, name='dense_'+str(i)))
    dense_layers.append(layers.Dense(out_shape, activation=activation, name='output'))

    if use_conv:
        if len(conv1_ls) == 1:
            a = conv1_ls[0](input_layer)
        else:
            cs = [c(input_layer) for c in conv1_ls]
            a = conv1_layer(cs)
        s = score_layer(a)
        s = soft_layer(s)
        a_s = a * tf.reshape(s, (-1, s.shape[-1], 1))
    else:
        s = score_layer(input_layer)
        s = soft_layer(s)
        i_s = input_layer * tf.reshape(s, (-1, s.shape[-1], 1))
        a_s = conv1_layer(i_s)
        
        if len(conv1_ls) == 1:
            a_s = conv1_ls[0](i_s)
        else:
            cs = [c(i_s) for c in conv1_ls]
            a_s = conv1_layer(cs)
    m = max_layer(a_s)
    for d in dense_layers:
        m = d(m)
    model = models.Model(input_layer, m)
    model.compile(loss=losses.BinaryCrossentropy(), optimizer=optimizers.Adam(), metrics=tms)
    return model