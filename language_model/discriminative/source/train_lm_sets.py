
import dynet as dy
import sys
import argparse
import time
import random
import numpy as np
from collections import defaultdict
import codecs
import math
import seaborn as sns
import pandas as pd
import pickle
import json
from LM_sets import LM_sets


parser = argparse.ArgumentParser()
parser.add_argument("--dynet-mem", help="allocate memory for dynet")
parser.add_argument("--dynet-gpu", help="use GPU")
parser.add_argument("--dynet-gpus", help="use GPU")
parser.add_argument("--dynet-seed", help="set random seed for dynet")
parser.add_argument("--dynet-gpu-ids", default=3, help="choose which GPU to use")
parser.add_argument("--dynet-autobatch", default=0, help="choose which GPU to use")
parser.add_argument("--dynet-devices", help="set random seed for dynet")
parser.add_argument("--dynet-weight-decay", help="choose weight decay")

parser.add_argument("--train", default="../../data/alternate_sents_train.json", help="location of training file")
parser.add_argument("--dev", default="../../data/alternate_sents_dev.json", help="location of validation file")
parser.add_argument("--test", default="../../data/alternate_sents_test.json", help="location of test file")
parser.add_argument("--train_finetune", default="../../data/alternate_sents_monolingual.json", help="location of training file for finetune")
parser.add_argument("--epochs", default=25, type=int, help="number of epochs")
parser.add_argument("--batch_size", default=20, type=int, help="size of batches")
parser.add_argument("--trainer", default="sgd", help="choose trainer for the optimization")
parser.add_argument("--num_layers", default=2, help="number of layers in RNN")
parser.add_argument("--input_dim", default=300, type=int, help="dimension of the input to the RNN")
parser.add_argument("--hidden_dim", default=650, type=int, help="dimension of the hidden layer of the RNN")
parser.add_argument("--x_dropout", default=0.35, type=float, help="value for rnn dropout")
parser.add_argument("--h_dropout", default=0.35, type=float, help="value for rnn dropout")
parser.add_argument("--w_dropout_rate", default=0.2, type=float, help="value for rnn dropout")
parser.add_argument("--logfile", default="log.txt", help="location of log file for debugging")
parser.add_argument("--learning_rate", default=1, type = float, help="set initial learning rate")
parser.add_argument("--lr_decay_factor", default=2.5, type = float, help="set clipping threshold")
parser.add_argument("--clip_thr", default=1, type=float, help="set clipping threshold")
parser.add_argument("--init_scale_rnn", default=0.05, type=float, help="scale to init rnn")
parser.add_argument("--init_scale_params", default=None, type=float, help="scale to init params")
parser.add_argument("--check_freq", default=None, help="frequency of checking perp on dev and updating lr")
parser.add_argument("--finetune_p1", action='store_true',help="train a model from monolingual data")
parser.add_argument("--finetune_p2", help="name of model learned from momnolingual data")
parser.add_argument("--lamb", default=1, type=float, help="value of lambda")
parser.add_argument("--margin", default="wer", help="value of margin for loss")


def wer(r, h):
    #taken from https://martin-thoma.com/word-error-rate-calculation/

    # initialisation
    r = r.split()
    h = h.split()
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    res = d[len(r)][len(h)]/float(len(r))
    assert(res > 0)
    return res


class Set:
    orig = None
    orig_type = None
    cs = None
    en = None
    sp = None
    distances = None


def remove_tag(l):

    no_tag = ""
    for w in l.split():
        no_tag += w.split("__")[0] + " "

    return no_tag


def detect_lang(sent):

    en = any([w.endswith("__en") for w in sent.split()])
    sp = any([w.endswith("__sp") for w in sent.split()])

    if en and sp:
        return 0
    if en:
        return 1
    if sp:
        return 2

    return None


def read_data(data_file, train = False):
    all_sets = []
    with open(data_file) as f:
        data = json.load(f)
    for s in data:
        new_s = Set()
        new_s.cs = []
        new_s.en = []
        new_s.sp = []
        new_s.distances = []
        new_s.orig_type = detect_lang(s["orig"])

        sent = []
        for w in [START_TOKEN] + s["orig"].split() + [END_TOKEN]:
            sent.append(w2i[w.split("__")[0].lower()])
        new_s.orig = sent
        for i, type in enumerate(["cs", "en", "sp"]):
            for v in s[type+"_alternatives"]:
                assert(detect_lang(v) == i)
                sent = []
                for w in [START_TOKEN] + v.split() + [END_TOKEN]:
                    sent.append(w2i[w.split("__")[0].lower()])
                getattr(new_s, type).append(sent)
                if args.margin == "wer":
                    new_s.distances.append(wer(s["orig"], v))
                else:
                    new_s.distances.append(float(args.margin))
        all_sets.append(new_s)
    assert(len(all_sets) == len(data))
    if train:
        print "length of train:", len(all_sets)
    return all_sets, len(all_sets)


def evaluate(test):

    ranks = []
    for item in test:
        batch = [item.orig] + item.cs + item.en + item.sp
        scores = [s.npvalue() for s in get_batch_scores(batch, True)]
        indices = list(range(len(scores)))
        indices.sort(key=lambda x: scores[x], reverse=True)
        ranks.append(indices.index(0)+1)

    cs = []
    mono = []
    for rank, item in zip(ranks, test):
        if item.orig_type:
            mono.append(rank)
        else:
            cs.append(rank)

    tot_acc = sum([1 for r in ranks if r == 1])/float(10)
    cs_acc = sum([1 for r in cs if r == 1])/float(2.5)
    mono_acc = sum([1 for r in mono if r == 1])/float(7.5)

    return tot_acc, cs_acc, mono_acc


def get_batch_scores(batch, evaluate = False):

    return model.get_batch_scores(batch, evaluate)[0]


def calc_loss(batch_distances, batch_scores, lamb):

    batch_losses = [lamb * dy.scalarInput(d) - (batch_scores[0] - s) for s, d in zip(batch_scores[1:], batch_distances[1:])]
    losses_pos = [l if l.npvalue() >= 0 else dy.scalarInput(0) for l in batch_losses]

    if len(losses_pos) == 0:
        return 0

    return dy.esum(losses_pos)


def logstr(f, s):

    print s
    f.write(s)


def check_performance(epoch, train_acc, best_acc, best_epoch, batch_n = None):

    if args.check_freq:
        logstr(f_log, "batch number {}. note: train acc is only on last {} batches\n\n".format(str(batch_n), args.check_freq))

    dev_acc, dev_cs_acc, dev_mono_acc = evaluate(dev_set)
    logstr(f_log, "dev_cs_acc {}\n".format(dev_cs_acc))
    logstr(f_log, "dev_mono_acc {}\n".format(dev_mono_acc))

    test_acc, test_cs_acc, test_mono_acc = evaluate(test_set)
    logstr(f_log, "test_cs_acc {}\n".format(test_cs_acc))
    logstr(f_log, "test_mono_acc {}\n".format(test_mono_acc))

    if dev_acc > best_acc:
        best_acc = dev_acc
        best_epoch = epoch
        if args.finetune_p1:
            model.save(args.logfile, vocab=w2i)
        else:
            model.save(args.logfile)
    else:
        model.update_lr(args.lr_decay_factor)

    logstr(f_log, "train_acc "+str(train_acc)+"\n")
    logstr(f_log, "dev_acc "+str(dev_acc)+"\n")
    logstr(f_log, "test_acc "+str(test_acc)+"\n")
    logstr(f_log, "best_so_far " + str(best_acc) + "\n")
    logstr(f_log, "learning_rate " + str(model.get_learning_rate()) + "\n\n")

    return dev_acc, best_acc, best_epoch


def train():

    train_time = time.time()
    best_acc = 0
    best_epoch = -1

    try:
        for epoch in range(args.epochs):
            epoch_time = time.time()
            random.shuffle(train_set)
            train_acc = 0
            train_loss = 0
            for i, t_set in enumerate(train_set):
                batch = [t_set.orig] + t_set.cs + t_set.en + t_set.sp
                if args.check_freq and i > 0 and i % int(args.check_freq) == 0:
                    train_acc = (train_acc/float(args.check_freq))*100
                    dev_acc, best_acc, best_epoch = check_performance(epoch, train_acc, best_acc, best_epoch, i)
                    train_acc = 0
                batch_scores = get_batch_scores(batch)
                if np.argmax([s.npvalue() for s in batch_scores]) == 0:
                    train_acc += 1
                batch_loss = calc_loss(t_set.distances, batch_scores, args.lamb)
                ll = batch_loss.npvalue()
                train_loss += ll
                if i> 0 and i % 500 == 0:
                    print "avg loss is", train_loss/float(500)
                    train_loss = 0
                batch_loss.backward()
                model.trainer_update()

            if not args.check_freq:
                train_acc = (train_acc/float(train_nbatches))*100
                dev_acc, best_acc, best_epoch = check_performance(epoch, train_acc, best_acc, best_epoch)

            logstr(f_log, "time for epoch number " + str(epoch) + " is: " + str(time.time() - epoch_time) + "\n\n")


    except KeyboardInterrupt:
        logstr(f_log, "Exiting from training early\n\n")

    logstr(f_log, "time for training is: " + str(time.time() - train_time) + "\n\n")

    # results
    logstr(f_log, "best acc is: " + str(best_acc) + "\n\n")
    logstr(f_log, "best epoch: " + str(best_epoch) + "\n\n")


if __name__ == '__main__':

    args = parser.parse_args()
    random.seed(10)
    np.random.seed(10)

    f_log = open("../models/" + args.logfile + ".txt", "w", 0)
    logstr(f_log, str(args) + "\n\n")

    START_TOKEN = "<s>"
    END_TOKEN = "</s>"
    w2i = defaultdict(lambda: len(w2i))
    UNK = w2i["<unk>"]

    train_set, train_nbatches = read_data(args.train, True)
    dev_set, dev_nbatches = read_data(args.dev)
    test_set, test_nbatches = read_data(args.test)

    if args.finetune_p1:

        # use the monolingual data instead of the cs data
        args.train = args.train_finetune
        train_set, train_nbatches = read_data(args.train, True)

    if args.finetune_p2:

        # the model needs to know the vocabulary of the monolingual data
        train_set_finetune, train_nbatches_finetune = read_data(args.train_finetune, True)


    # create model
    word_num = len(w2i)
    model = LM_sets(args.num_layers, args.input_dim, args.hidden_dim, word_num, args.init_scale_rnn, args.init_scale_params,
               args.x_dropout, args.h_dropout, args.w_dropout_rate, args.learning_rate, args.clip_thr)

    if args.finetune_p2:
        # load pre-trained model
        model.load("../models/" + args.finetune_p2 + "_model")
        print "loaded model", args.finetune_p2


    train()

