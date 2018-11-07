
import os
import math
import sys
import subprocess
import string
import random
import numpy as np
import pdb
import time
import json
import argparse
from multiprocessing import Pool
from functools import partial
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--output_file", help="name of created file")
parser.add_argument("--input_file", help="name of input file")
parser.add_argument("--l1", default="en", help="name of language 1")
parser.add_argument("--l2", default="sp", help="name of language 2")
parser.add_argument("--l1_l2_fst", default="../../data/FSTs/ensp", help="location of FST for L1+L2")
parser.add_argument("--l1_fst_inv", default="../../data/FSTs/en_inv", help="location of FST_inv for L1")
parser.add_argument("--l2_fst_inv", default="../../data/FSTs/sp_inv", help="location of FST_inv for L2")
parser.add_argument("--l1_l2_fst_inv", default="../../data/FSTs/ensp_inv", help="location of FST_inv for L1+L2")
parser.add_argument("--change_fst", default="../../data/FSTs/fst_change_phones", help="location of FST for changing phonemes")
parser.add_argument("--probs_l1", default="../../data/probs/probs_en.txt", help="location of probs file for language 1")
parser.add_argument("--probs_l2", default="../../data/probs/probs_sp.txt", help="location of probs file for language 2")
parser.add_argument("--min_length", default=4, type=int, help="min length of sampled text")
parser.add_argument("--max_length", default=8, type=int, help="max length of sampled text")
parser.add_argument("--pool", type=int, help="num of threads")
args = parser.parse_args()


def avg_len_words(sent):

    s_split = sent.split()
    len_l1 = len_l2 = num_l1 = num_l2 = 0
    for w in s_split:
        if "__" not in w:
            continue
        w_s =  w.split("__")
        if w_s[1] == args.l1:
            len_l1 += len(w_s[0])
            num_l1 += 1
        else:
            assert(w_s[1] == args.l2)
            len_l2 += len(w_s[0])
            num_l2 += 1
    avg_l1 = 0
    avg_l2 = 0
    if num_l1 != 0:
        avg_l1 = len_l1/float(num_l1)
    if num_l2 != 0:
        avg_l2 = len_l2/float(num_l2)
    return avg_l1, avg_l2


# return how many words we have from each language
def count_per_lang(sent):

    s_split = sent.split()
    c = Counter([w.endswith("__" + args.l1) for w in s_split])
    return c[True] + c[False], c[True], c[False]


def diff_from_orig(orig, b):

    b = set(b.split())
    recall = len(set.intersection(orig, b))/float(len(orig))
    precision = len(set.intersection(orig, b))/float(len(b))

    return - recall - precision


def is_cs(s):

    return any([w.endswith("__" + args.l1) for w in s.strip().split()]) and any([w.endswith("__" + args.l2) for w in s.strip().split()])


def extract_vocabulary(probs_file):

    d = {}
    with open(probs_file, "r") as f:
        for l in f:
            l = l.split("\t")
            if float(l[1]) > 0:
                d[l[0]] = l[1]
    return d


def mono_alternatives(f, l_full, last_token):

    all_options = []
    alternatives = []
    for s in f:
        # make sure the alternatives are different from the original and from each other
        if s.strip() and s not in all_options and s.strip() != l_full.strip():
            all_options.append(s)
            if len(all_options) == 10:
                break

    if not all_options:
        return []

    for s in all_options[:10]:
        print s.strip()
        alternatives.append(s.strip() + " " + last_token)

    return alternatives


def scores(l_partial, all_options):

    all_scores = []
    tot_orig, l1_orig, l2_orig = count_per_lang(l_partial)
    l1_dominant = 0
    l2_dominant = 0
    if l1_orig > l2_orig:
        l1_dominant = 1
    else:
        l2_dominant = 1
    for s in all_options:
        avg_l1, avg_l2 = avg_len_words(s)
        tot, l1, l2 = count_per_lang(s)
        tot_diff, l1_diff, l2_diff = tot_orig - tot, l1_orig - l1, l2_orig - l2
        all_scores.append( - tot_diff + l1_dominant*(l1_diff - l2_diff + 3*avg_l2) + l2_dominant*(l2_diff - l1_diff + 3*avg_l1))

    return all_scores


def choose_cs(f, l, l_partial, samp_idx, samp_len):

    all_cs = []
    all_options = []
    for s in f:
        # make sure the alternatives are different from the original and are indeed cs
        if s.strip() and s.strip() not in all_options and s.strip() != l_partial.strip() and is_cs(" ".join(l[:samp_idx] + s.split() + l[samp_idx+samp_len:])):
            all_options.append(s)

    if len(all_options) == 0:
        return []

    # give a score to each option (we want them to be good - we use some heuristics)
    all_scores = scores(l_partial, all_options)

    # choose the 50 options with highest scores
    len_list = min(len(all_scores), 50)
    # this takes the best #len_list indecies to be at the end of the array, and then takes only those to be in best_idx
    best_idx = np.argpartition(all_scores, -len_list)[-len_list:]

    # then choose the best 10 that are as different as possible from the orig
    bests = [all_options[idx] for idx in best_idx]
    orig = set(l_partial.split())

    all_scores = []
    for b in bests:
        all_scores.append(diff_from_orig(orig, b))

    len_list = min(len(all_scores), 10)
    best_idx = np.argpartition(all_scores, -len_list)[-len_list:]
    for idx in best_idx:
        b = bests[idx]
        print b.strip()
        all_cs.append(" ".join(l[:samp_idx] + b.split() + l[samp_idx+samp_len:]))

    return all_cs


def upper(line):

    l = ""
    for w_tag in line.split():
        if "__" in w_tag:
            w, tag = w_tag.split("__")
            w_tag = w.upper() + "__" + tag
            l += w_tag + " "
        else:
            l += w_tag.upper() + " "

    return l


def check_in_vocab(l, vocab_l1, vocab_l2):

    for w in l:
        if (w.endswith("__"+args.l1) and w.split("__")[0] not in vocab_l1) or (w.endswith("__"+args.l2) and w.split("__")[0] not in vocab_l2):
            return False

    return True


def main(fname, lines, iter = None):

    # for reproducibility
    random.seed(10)

    if iter:
        fname = fname + str(iter[0]+1)
        s_idx = iter[1][0]
        e_idx = iter[1][1]
        lines = lines[s_idx : e_idx]
    else:
        s_idx = 0

    punct = list(string.punctuation) + ["..."]
    vocab_l1 = extract_vocabulary(args.probs_l1)
    vocab_l2 = extract_vocabulary(args.probs_l2)

    cs_file = "cs_file_{}".format(fname)
    l1_file = args.l1 + "_file_{}".format(fname)
    l2_file = args.l2 + "_file_{}".format(fname)

    all_sets = []

    line_num = s_idx

    for line in lines:

        line_num += 1
        print "line",  line_num

        l = upper(line).split()
        length = len(l)
        if l[-1] in punct:
            length -= 1

        # sets the range of length of the segment to be changed
        MIN = args.min_length
        MAX = min(args.max_length, length)

        if MAX < MIN:
            MIN = MAX

        if l[-1] in punct:
            l_full = l[:-1]
            last_token = l[-1]
        else:
            l_full = l
            last_token = ""

        if any(x.split("__")[0] in punct for x in l_full):
            l_full = [x for x in l_full if x.split("__")[0] not in punct]

        l_full = " ".join(l_full)

        os.system("echo  \" {} \"  | carmel  -sliOQWEk 100 {} {} {} > {}".format(l_full, args.l1_l2_fst, args.change_fst, args.l1_fst_inv, l1_file))
        os.system("echo  \" {} \"  | carmel  -sliOQWEk 100 {} {} {} > {}".format(l_full, args.l1_l2_fst, args.change_fst, args.l2_fst_inv, l2_file))

        for _ in range(5):

            samp_len = random.sample(range(MIN, MAX+1), 1)[0]
            samp_idx = random.sample(range(length-samp_len+1), 1)[0]
            l_partial = " ".join(l[samp_idx:samp_idx+samp_len])

            # make sure all words appear in the corresponding vocabulary
            if check_in_vocab(l[samp_idx:samp_idx+samp_len], vocab_l1, vocab_l2):
                os.system("echo  \" {} \"  | carmel  -sliOQWEk 1000 {} {} {} > {}".format(l_partial, args.l1_l2_fst, args.change_fst, args.l1_l2_fst_inv, cs_file))

                with open(cs_file) as f_cs:
                    if all(not l.strip() for l in f_cs):
                        # we found no cs alternative, so we try again with a different segment
                        continue
                    else:
                        break

            else:
                os.system("echo  \" \"  > {}".format(cs_file))
                continue

        with open(cs_file) as f:
            cs_alternatives = choose_cs(f, l, l_partial, samp_idx, samp_len)

        with open(l1_file) as f:
            l1_alternatives = mono_alternatives(f, l_full, last_token)

        with open(l2_file) as f:
            l2_alternatives = mono_alternatives(f, l_full, last_token)

        orig = " ".join(l)
        all_sets.append({"orig": orig, "cs_alternatives": cs_alternatives, args.l1+"_alternatives": l1_alternatives, args.l2+"_alternatives": l2_alternatives})


    with open("../../data/alternatives/alternate_sents_{}.json".format(fname), "w") as f_alternate:
        json.dump(all_sets, f_alternate)

    return line_num - s_idx


if __name__ == '__main__':

    fname = args.output_file
    with open(args.input_file) as f_input:
        lines = f_input.readlines()

    tot_sents = 0
    if args.pool:
        pool = Pool(args.pool)
        b_size = int(math.ceil(float(len(lines))/args.pool))
        starts = range(0,len(lines),b_size)
        ends = [s+b_size for s in starts]
        assert(len(starts) <= args.pool)

        func = partial(main, fname, lines)
        for p in pool.imap_unordered(func, enumerate(zip(starts, ends))):
            tot_sents += p

        all_sets = []
        for i in range(len(starts)):
            with open("../../data/alternatives/alternate_sents_{}{}.json".format(fname, str(i+1)), "r") as f:
                all_sets += json.load(f)
            os.remove("../../data/alternatives/alternate_sents_{}{}.json".format(fname, str(i+1)))
            for t in ["cs", args.l1, args.l2]:
                os.remove("{}_file_{}{}".format(t, fname, str(i+1)))
        with open("../../data/alternatives/alternate_sents_{}.json".format(fname), "w") as f_alternate:
            json.dump(all_sets, f_alternate)

    else:
        tot_sents = main(fname, lines)
        for t in ["cs", args.l1, args.l2]:
            os.remove("{}_file_{}".format(t, fname))

    print "tot_sents", tot_sents

