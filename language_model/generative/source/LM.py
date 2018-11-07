
import dynet as dy
import numpy as np
import random

class LM(object):

    def __init__(self, num_layers, input_dim, hidden_dim, word_num, init_scale_rnn, init_scale_params, x_dropout, h_dropout, w_dropout_rate, lr, clip_thr):

        model = dy.Model()

        rnn = dy.LSTMBuilder(num_layers, input_dim, hidden_dim, model)
        self.init_rnn(rnn, init_scale_rnn)

        params = {}
        params["embeds"] = model.add_lookup_parameters((word_num, input_dim))
        params["b_p"] = model.add_parameters((word_num,))
        params["W_p"] = model.add_parameters((word_num, hidden_dim))
        if init_scale_params:
            self.init_lookup(embeds, init_scale_params)
            self.init_param(params["W_p"], init_scale_params)
            self.init_param(params["b_p"], 0)

        trainer = dy.SimpleSGDTrainer(model, lr)
        if clip_thr >0:
            trainer.set_clip_threshold(clip_thr)

        self._model = model
        self._rnn = rnn
        self._params = params
        self._x_dropout = x_dropout
        self._h_dropout = h_dropout
        self._w_dropout_rate = w_dropout_rate
        self._trainer = trainer
        self._input_dim = input_dim

    def init_param(self, param, scale):
        dims = param.as_array().shape
        param.set_value(2*scale*np.random.rand(*dims) - scale)

    def init_lookup(self, param, scale):
        dims = param.as_array().shape
        param.init_from_array(2*scale*np.random.rand(*dims) - scale)

    def init_rnn(self, rnn, init_scale_rnn):
        pc = rnn.param_collection()
        pl = pc.parameters_list()
        for p in pl:
            dims = p.as_array().shape
            p.set_value(2*init_scale_rnn*np.random.rand(*dims) - init_scale_rnn)
        return

    def word_dropout(self, seq, w_dropout_rate):

        w_dropout = []
        for w in set(seq):
            p = random.random()
            if p < w_dropout_rate:
                w_dropout.append(w)

        return w_dropout

    def update_lr(self, lr_decay_factor):

        self._trainer.learning_rate /= lr_decay_factor

        return

    def trainer_update(self):

        self._trainer.update()

    def get_learning_rate(self):

        return self._trainer.learning_rate

    def get_loss(self, seq, evaluate = False):

        if evaluate:
            self._rnn.disable_dropout()
        else:
            self._rnn.set_dropouts(self._x_dropout, self._h_dropout)
        state = self._rnn.initial_state()

        W = dy.parameter(self._params["W_p"])
        b = dy.parameter(self._params["b_p"])

        # use word dropout when training
        dropped = dy.inputTensor(np.zeros(self._input_dim))
        if evaluate:
            vecs = [self._params["embeds"][w] for w in seq]
        else:
            w_dropout = self.word_dropout(seq, self._w_dropout_rate)
            vecs = [self._params["embeds"][w] if w not in w_dropout else dropped for w in seq]

        outputs = [x.output() for x in state.add_inputs(vecs[:-1])]

        seq_losses = []
        for idx, output in enumerate(outputs):
            y = dy.affine_transform([b, W, output])
            loss = dy.pickneglogsoftmax(y, seq[idx+1])
            seq_losses.append(loss)

        return dy.esum(seq_losses)/(len(seq)-1)

    def save(self, logfile, vocab = None, model_type = None):

        if model_type == "rank":
            self._model.save("../models/" + logfile + "_model_rank")
            return

        self._model.save("../models/" + logfile + "_model")
        if vocab:
            with open("../models/" + logfile + "_vocab", "w") as f:
                for w in vocab:
                    f.write(w.encode("utf-8") + " " + str(vocab[w]) +"\n")

    def load(self, model_to_load):

        print "in func, loaded model"
        self._model.populate(model_to_load)




