
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
from LM import LM


parser = argparse.ArgumentParser()
parser.add_argument("--dynet-mem", help="allocate memory for dynet")
parser.add_argument("--dynet-gpu", help="use GPU")
parser.add_argument("--dynet-gpus", help="use GPU")
parser.add_argument("--dynet-seed", help="set random seed for dynet")
parser.add_argument("--dynet-gpu-ids", default=3, help="choose which GPU to use")
parser.add_argument("--dynet-autobatch", default=0, help="choose which GPU to use")
parser.add_argument("--dynet-devices", help="set random seed for dynet")
parser.add_argument("--dynet-weight-decay", help="choose weight decay")

parser.add_argument("--train", default="../../data/bangor_train_gold", help="location of training file")
parser.add_argument("--dev", default="../../data/bangor_dev_gold", help="location of validation file")
parser.add_argument("--test", default="../../data/bangor_test_gold", help="location of test file")
parser.add_argument("--new_dev_file", default="../../data/alternate_sents_dev.json", help="new dev file")
parser.add_argument("--new_test_file", default="../../data/alternate_sents_test.json", help="new test file")
parser.add_argument("--train_finetune", default="../../data/mono_train_gold", help="location of training file for finetune")
parser.add_argument("--dev_finetune", default="../../data/mono_dev_gold", help="location of dev file for finetune")
parser.add_argument("--test_finetune", default="../../data/mono_test_gold", help="location of test file for finetune")
parser.add_argument("--new_test", action='store_true', help="evaluate on my test as well")
parser.add_argument("--epochs", default=40, type=int, help="number of epochs")
parser.add_argument("--batch_size", default=20, type=int, help="size of batches")
parser.add_argument("--trainer", default="sgd", help="choose trainer for the optimization")
parser.add_argument("--num_layers", default=2, help="number of layers in RNN")
parser.add_argument("--input_dim", default=300, type=int, help="dimension of the input to the RNN")
parser.add_argument("--hidden_dim", default=650, type=int, help="dimension of the hidden layer of the RNN")
parser.add_argument("--x_dropout", default=0.35, type=float, help="value for rnn dropout")
parser.add_argument("--h_dropout", default=0.35, type=float, help="value for rnn dropout")
parser.add_argument("--w_dropout_rate", default=0.2, type=float, help="value for rnn dropout")
parser.add_argument("--logfile", default="log", help="name of log file")
parser.add_argument("--learning_rate", default=10, type = float, help="set initial learning rate")
parser.add_argument("--lr_decay_factor", default=2.5, type = float, help="set clipping threshold")
parser.add_argument("--clip_thr", default=1, type=float, help="set clipping threshold")
parser.add_argument("--init_scale_rnn", default=0.05, type=float, help="scale to init rnn")
parser.add_argument("--init_scale_params", default=None, type=float, help="scale to init params")
parser.add_argument("--check_freq", default=None, help="frequency of checking perp on dev and updating lr")
parser.add_argument("--finetune_p1", action='store_true',help="train a model from monolingual data")
parser.add_argument("--finetune_p2", help="name of model learned from momnolingual data")
parser.add_argument("--load_model", help="model to load and evaluate, no training")

class Test_item:
    orig = None
    orig_type = None
    cs = None
    en = None
    sp = None


def read_corpus(raw_file):

    data = []
    f = open(raw_file)
    for l in f:
        new_s = []
        if l.strip(): l = l.split()
        else: continue
        for w in [START_TOKEN] + l + [END_TOKEN]:
            new_s.append(w2i[unicode(w.split("__")[0], "utf-8").lower()])
        data.append(new_s)

    # divide into batches
    data = [data[i:i+args.batch_size] for i in range(0, len(data), args.batch_size)]

    return data, len(data)


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


def read_testset(test_file):

    all_sets = []
    with open(test_file) as f:
        test = json.load(f)

    for t in test:

        new_t = Test_item()
        new_t.cs = []
        new_t.en = []
        new_t.sp = []

        new_t.orig_type = detect_lang(t["orig"])
        sent = []
        for w in [START_TOKEN] + t["orig"].split() + [END_TOKEN]:
            sent.append(w2i[(w.split("__")[0]).lower()])
        new_t.orig = sent

        for i, type in enumerate(["cs", "en", "sp"]):
            for v in t[type+"_alternatives"]:
                assert(detect_lang(v) == i)
                sent = []
                for w in [START_TOKEN] + v.split() + [END_TOKEN]:
                    sent.append(w2i[(w.split("__")[0]).lower()])
                getattr(new_t, type).append(sent)
        all_sets.append(new_t)

    assert(len(all_sets) == len(test))

    return all_sets


# evaluate with the new evaluation method + dataset
def eval_new_test(test):

    ranks = []
    for item in test:
        evals = []
        for s in [item.orig] + item.cs + item.en + item.sp:
            evals.append(calc_perp([[s]]))
        indices = list(range(len(evals)))
        indices.sort(key=lambda x: evals[x])
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


def calc_perp(data_set):

    perp = 0
    for batch in data_set:

        batch_loss = get_batch_loss(batch, True)
        perp += batch_loss.npvalue()

    return np.exp(perp/float(len(data_set)))


def get_batch_loss(batch, evaluate = False):

    dy.renew_cg()
    batch_losses = []

    for seq in batch:
        batch_losses.append(model.get_loss(seq, evaluate))

    return dy.esum(batch_losses)/len(batch)


def logstr(f, s):

    print s
    f.write(s)


def check_performance(epoch, train_perp, best_perp, best_epochs, best_acc, batch_n = None):

    if args.check_freq:
        logstr(f_log, "batch number {}. note: train perp is only on last {} batches\n\n".format(str(batch_n), args.check_freq))

    dev_perp = calc_perp(dev_set)
    test_perp = calc_perp(test_set)

    if dev_perp < best_perp:
        best_perp = dev_perp
        best_epochs[0] = epoch
        if args.finetune_p1:
            model.save(args.logfile, vocab=w2i)
        else:
            model.save(args.logfile)
    else:
        model.update_lr(args.lr_decay_factor)

    if args.new_test:
        dev_acc, dev_cs_acc, dev_mono_acc = eval_new_test(new_dev)
        logstr(f_log, "dev_cs_acc {}\n".format(dev_cs_acc))
        logstr(f_log, "dev_mono_acc {}\n".format(dev_mono_acc))

        test_acc, test_cs_acc, test_mono_acc = eval_new_test(new_test)
        logstr(f_log, "test_cs_acc {}\n".format(test_cs_acc))
        logstr(f_log, "test_mono_acc {}\n".format(test_mono_acc))

        if dev_acc > best_acc:
            best_acc = dev_acc
            best_epochs[1] = epoch
            model.save(args.logfile, model_type="rank")

    logstr(f_log, "train_perp "+str(train_perp)+"\n")
    logstr(f_log, "dev_perp "+str(dev_perp)+"\n")
    logstr(f_log, "test_perp "+str(test_perp)+"\n")
    logstr(f_log, "best_so_far " + str(best_acc) + "\n")
    logstr(f_log, "learning_rate " + str(model.get_learning_rate()) + "\n\n")

    if args.new_test:
        logstr(f_log, "dev_acc " + str(dev_acc) + "\n")
        logstr(f_log, "test_acc " + str(test_acc) + "\n\n")

    return dev_perp, best_perp, best_epochs, best_acc


def train(train_set):

    train_time = time.time()
    best_acc = 0
    best_perp = float("inf")
    best_epochs = [-1, -1]

    try:
        for epoch in range(args.epochs):

            epoch_time = time.time()

            random.shuffle(train_set)

            train_perp = 0
            for i,batch in enumerate(train_set):
                if args.check_freq and i > 0 and i % int(args.check_freq) == 0:
                    train_perp = np.exp(train_perp/float(args.check_freq))
                    dev_perp, best_perp, best_epochs, best_acc = check_performance(epoch, train_perp, best_perp, best_epochs, best_acc, i)
                    train_perp = 0

                batch_loss = get_batch_loss(batch)
                train_perp += batch_loss.npvalue()
                batch_loss.backward()
                model.trainer_update()

            if not args.check_freq:
                train_perp = np.exp(train_perp/float(train_nbatches))
                dev_perp, best_perp, best_epochs, best_acc = check_performance(epoch, train_perp, best_perp, best_epochs, best_acc)

            logstr(f_log, "time for epoch number " + str(epoch) + " is: " + str(time.time() - epoch_time) + "\n\n")


    except KeyboardInterrupt:
        logstr(f_log, "Exiting from training early\n\n")

    logstr(f_log, "time for training is: " + str(time.time() - train_time) + "\n\n")

    # results
    logstr(f_log, "best perp is: " + str(best_perp) + "\n\n")
    if args.new_test:
        logstr(f_log, "best_acc is: " + str(best_acc) + "\n\n")
        logstr(f_log, "corresponding epochs: " + str(best_epochs) + "\n\n")


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

    train_set, train_nbatches = read_corpus(args.train)
    dev_set, dev_nbatches = read_corpus(args.dev)
    test_set, test_nbatches = read_corpus(args.test)


    if args.finetune_p1:

        # use the monolingual data instead of the cs data
        args.train = args.train_finetune
        args.dev = args.dev_finetune
        args.test = args.test_finetune

        train_set, train_nbatches = read_corpus(args.train)
        dev_set, dev_nbatches = read_corpus(args.dev)
        test_set, test_nbatches = read_corpus(args.test)


    if args.finetune_p2:

        # the model needs to know the vocabulary of the monolingual data

        train_set_finetune, train_nbatches_finetune = read_corpus(args.train_finetune)
        dev_set_finetune, dev_nbatches_finetune = read_corpus(args.dev_finetune)
        test_set_finetune, test_nbatches_finetune = read_corpus(args.test_finetune)


    if args.new_test:
        new_dev = read_testset(args.new_dev_file)
        new_test = read_testset(args.new_test_file)

    # create model
    word_num = len(w2i)
    model = LM(args.num_layers, args.input_dim, args.hidden_dim, word_num, args.init_scale_rnn, args.init_scale_params,
               args.x_dropout, args.h_dropout, args.w_dropout_rate, args.learning_rate, args.clip_thr)

    if args.finetune_p2 and not args.load_model:
        # load pre-trained model
        model.load("../models/" + args.finetune_p2 + "_model")
        print "loaded model", args.finetune_p2

    if args.load_model:
        # load pre-trained model
        model.load("../models/" + args.load_model + "_model")
        print "loaded model", args.load_model
        args.dev = "../../../evaluation_dataset/raw/bangor_dev_gold"
        args.test = "../../../evaluation_dataset/raw/bangor_test_gold"
        dev_set, dev_nbatches = read_corpus(args.dev)
        test_set, test_nbatches = read_corpus(args.test)
        dev_perp = calc_perp(dev_set)
        test_perp = calc_perp(test_set)
        dev_acc, dev_cs_acc, dev_mono_acc = eval_new_test(new_dev)
        test_acc, test_cs_acc, test_mono_acc = eval_new_test(new_test)
        print "dev_perp", dev_perp
        print "test_perp", test_perp
        print "dev accs", dev_acc, dev_cs_acc, dev_mono_acc
        print "test accs", test_acc, test_cs_acc, test_mono_acc

        exit()

    train(train_set)
