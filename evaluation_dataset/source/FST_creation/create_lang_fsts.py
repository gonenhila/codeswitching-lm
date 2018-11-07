
import string
import itertools
from operator import add
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--l1", default="en", help="name of language 1")
parser.add_argument("--l2", default="sp", help="name of language 2")
parser.add_argument("--probs_l1", default="../../data/probs/probs_en.txt", help="location of probs file for language 1")
parser.add_argument("--probs_l2", default="../../data/probs/probs_sp.txt", help="location of probs file for language 2")
parser.add_argument("--dict_l1", default="../../data/dictionaries/dict_en", help="location of dictionary file for language 1")
parser.add_argument("--dict_l2", default="../../data/dictionaries/dict_sp", help="location of dictionary file for language 2")
parser.add_argument("--single_char_l1", default="AI", help="a string with characters that are o.k in L1 on there own")
parser.add_argument("--single_char_l2", default="AEOUY", help="a string with characters that are o.k in L2 on there own")
parser.add_argument("--FST", default="../../data/FSTs/", help="location of created FSTs")
parser.add_argument("--mappings", default="../../data/mappings/mappings.json", help="location of mappings file")
args = parser.parse_args()


# extract unigram probabilities
def extract_unigrams(filename):

    d_probs = {}
    tot = 0
    for l in open(filename):
        w, p = l.split("\t")
        d_probs[w] = float(p)
        tot += float(p)
    for w in d_probs:
        d_probs[w] = d_probs[w]/tot

    return d_probs


# make the additional changes to make both sets as similar as possible
def replace_phones(v, mapping):

    v_all = []
    v = v.split()
    for p in v:
        if p in mapping:
            v_all.append(mapping[p])
        else:
            v_all.append(p)
    return " ".join(v_all)


# create a python dictionary from the phonemes dictionary
def create_dict(map_phones, dict):

    d = {}
    for l in open(dict):
        k, v = l.strip().split("\t", 1)
        k = k.upper()
        v = v.split("\t")

        d[k] = []
        for item in v:
            d[k].append(replace_phones(item, map_phones))

    return d

def write_to_files(i, diff, v, k , d_probs, f, f_inv, f_both, f_both_inv, lang):

    if len(v) == 1:
        f.write('(0 (0 {}__{} {} {}))\n'.format(k, lang, v[0], d_probs[k]))
        f_both.write('(0 (0 {}__{} {} {}))\n'.format(k, lang, v[0], d_probs[k]))
        f_inv.write('(0 (0 {} {}__{} {}))\n'.format(v[0], k, lang, d_probs[k]))
        f_both_inv.write('(0 (0 {} {}__{} {}))\n'.format(v[0], k, lang, d_probs[k]))
    if len(v) > 1:
        l = len(v)
        f.write('(0 ({} *e* {} {}))\n'.format(i+1, v[0], d_probs[k]))
        f_both.write('(0 ({} *e* {} {}))\n'.format(i+diff+1, v[0], d_probs[k]))
        f_inv.write('(0 ({} {} *e* {}))\n'.format(i+1, v[0], d_probs[k]))
        f_both_inv.write('(0 ({} {} *e* {}))\n'.format(i+diff+1, v[0], d_probs[k]))

        f.write('({} (0 {}__{} {}))\n'.format(i+l-1, k, lang, v[l-1]))
        f_both.write('({} (0 {}__{} {}))\n'.format(i+diff+l-1, k, lang, v[l-1]))
        f_inv.write('({} (0 {} {}__{}))\n'.format(i+l-1, v[l-1], k, lang))
        f_both_inv.write('({} (0 {} {}__{}))\n'.format(i+diff+l-1, v[l-1], k, lang))
        for j,syl in enumerate(v[1:-1]):
            f.write('({} ({} *e* {}))\n'.format(i+j+1, i+j+2, syl))
            f_both.write('({} ({} *e* {}))\n'.format(i+diff+j+1, i+diff+j+2, syl))
            f_inv.write('({} ({} {} *e*))\n'.format(i+j+1, i+j+2, syl))
            f_both_inv.write('({} ({} {} *e*))\n'.format(i+diff+j+1, i+diff+j+2, syl))
        i = i + l - 1

    return i

def write_lang_to_file(i, diff, d, d_probs, f, f_inv, f_l1_l2, f_l1_l2_inv, lang):

    for k in d:
        if d_probs[k] == 0:
            continue
        for v in d[k]:
            v = v.split()
            i = write_to_files(i, diff, v, k , d_probs, f, f_inv, f_l1_l2, f_l1_l2_inv, lang)

    return i

# creates a file for FST in carmel
# This creates the FSTs from the dictionaries: l1, l2, l1+l2, and the inverted ones
# Each has edges with words, and it outputs the matching sequences of phones when a word is read (each phone on a separate edge)
# The inverted ones are opposite
def create_fsts(d_l1, d_l2, d_probs_l1, d_probs_l2):

    with open(args.FST+args.l1, "w") as f_l1, open(args.FST+args.l1+"_inv", "w") as f_l1_inv,          \
         open(args.FST+args.l2, "w") as f_l2, open(args.FST+args.l2+"_inv", "w") as f_l2_inv,          \
         open(args.FST+args.l1+args.l2, "w") as f_l1_l2, open(args.FST+args.l1+args.l2+"_inv", "w") as f_l1_l2_inv:

        f_l1.write("%%%% fst with separate phones from L1 dictionary %%%%\n0\n")
        f_l1_inv.write("%%%% fst with separate phones from L1 dictionary - inverted %%%%\n0\n")
        f_l2.write("%%%% fst with separate phones from L2 dictionary %%%%\n0\n")
        f_l2_inv.write("%%%% fst with separate phones from L2 dictionary - inverted %%%%\n0\n")
        f_l1_l2.write("%%%% fst with separate phones from L1+L2 dictionaries %%%%\n0\n")
        f_l1_l2_inv.write("%%%% fst with separate phones from L1+L2 dictionaries - inverted %%%%\n0\n")

        diff = write_lang_to_file(0, 0, d_l1, d_probs_l1, f_l1, f_l1_inv, f_l1_l2, f_l1_l2_inv, args.l1)
        diff = write_lang_to_file(0, diff, d_l2, d_probs_l2, f_l2, f_l2_inv, f_l1_l2, f_l1_l2_inv, args.l2)


if __name__ == '__main__':

    # extract unigram probabilities
    d_probs_l1 = extract_unigrams(args.probs_l1)
    d_probs_l2 = extract_unigrams(args.probs_l2)

    # discard words than end with "." or with ")"
    # discard words with one letter, except for a predefined list
    for w in d_probs_l1:
        if w.endswith(")") or w.endswith("."):
            d_probs_l1[w] = 0
        if len(w) == 1 and w not in args.single_char_l1:
            d_probs_l1[w] = 0

    for w in d_probs_l2:
        if w.endswith(")") or w.endswith("."):
            d_probs_l2[w] = 0
        if len(w) == 1 and w not in args.single_char_l2:
            d_probs_l2[w] = 0

    if args.l1 == "en" and args.l2 == "sp":
        with open(args.mappings, "r") as f:
            mappings = json.load(f)
        l2_l1_map, map_phones_l1, map_phones_l2 = mappings["l2_l1_map"], mappings["map_phones_l1"], mappings["map_phones_l2"]
    else:
        map_phones_l1 = map_phones_l2 = None

    d_l1 = create_dict(map_phones_l1, args.dict_l1)
    d_l2 = create_dict(map_phones_l2, args.dict_l2)

    create_fsts(d_l1, d_l2, d_probs_l1, d_probs_l2)
