
import json
import argparse
import pdb
import random

random.seed(10)

parser = argparse.ArgumentParser()
parser.add_argument("--fname", help="name of input file")
parser.add_argument("--cs", type=int, default=250, help="cs sentences required")
parser.add_argument("--mono", type=int, default=750 , help="monolingual sentences required")
parser.add_argument("--eval", action="store_true", help="whether this is a dev/test file, then we narrow done to #cs+#mono alternatives")
parser.add_argument("--min_orig_len", default=4, help="min length of original sentence")
parser.add_argument("--min_num_alter", default=5, help="min number of alternatives of each type")

args = parser.parse_args()

with open("../../data/alternatives/alternate_sents_{}.json".format(args.fname), "r") as f:
    all_sets = json.load(f)


def print_stats():
    print "total sets:"
    print len(all_sets)
    en_num = sp_num = cs_num = orig_cs = no_cs = no_en = no_sp = 0

    for s in all_sets:
        en_num += len(s["en_alternatives"])
        sp_num += len(s["sp_alternatives"])
        cs_num += len(s["cs_alternatives"])

        en = any([w.endswith("__en") for w in s["orig"].split()])
        sp = any([w.endswith("__sp") for w in s["orig"].split()])
        if en and sp:
            orig_cs += 1

        for sent in s["cs_alternatives"]:
            en = any([w.endswith("__en") for w in sent.split()])
            sp = any([w.endswith("__sp") for w in sent.split()])
            assert(en and sp)

        for sent in s["en_alternatives"]:
            assert(all([not w.endswith("__sp") for w in sent.split()]))

        for sent in s["sp_alternatives"]:
            assert(all([not w.endswith("__en") for w in sent.split()]))

        if len(s["cs_alternatives"]) == 0:
            no_cs += 1
        if len(s["en_alternatives"]) == 0:
            no_en += 1
        if len(s["sp_alternatives"]) == 0:
            no_sp += 1

    print "en:", en_num
    print "sp:", sp_num
    print "cs:", cs_num
    print "orig is cs", orig_cs
    print "no_cs", no_cs
    print "no_en", no_en
    print "no_sp", no_sp

print_stats()

# filter out sets (orig too short, not enough alternatives)
all_sets_filtered = []
for s in all_sets:
    if len(s["orig"].split()) < args.min_orig_len:
        continue
    if len(s["cs_alternatives"]) < args.min_num_alter or len(s["en_alternatives"]) < args.min_num_alter or len(s["sp_alternatives"]) < args.min_num_alter:
        continue
    all_sets_filtered.append(s)
all_sets = all_sets_filtered

# randomly choose sets with cs original sentence and sets with monolingual original sentence as required
if args.eval:
    tot = args.cs + args.mono
    sets_cs = []
    sets_mono = []
    random.shuffle(all_sets)
    for s in all_sets:
        en_w = sp_w = 0
        en = any([w.endswith("__en") for w in s["orig"].split()])
        sp = any([w.endswith("__sp") for w in s["orig"].split()])
        if en and sp and len(sets_cs) < args.cs:
            sets_cs.append(s)
            for w in s["orig"].split():
                if w.endswith("__en"):
                    en_w +=1
                if w.endswith("__sp"):
                    sp_w +=1

        if ((not en) or (not sp)) and len(sets_mono) < args.mono:
            sets_mono.append(s)

        if len(sets_cs) == args.cs and len(sets_mono) == args.mono:
            break

    all_sets = sets_cs + sets_mono
    random.shuffle(all_sets)
    assert(len(all_sets) == tot)

print "final"
print_stats()

with open("../../data/alternatives/alternate_sents_{}_filtered.json".format(args.fname), "w") as f:
    json.dump(all_sets, f)

# write sentences to a readable file
with open("../../data/alternatives/alternate_sents_{}_filtered_readable.txt".format(args.fname), "w") as f_view:
    for s in all_sets:
        f_view.write("orig:\n" + s["orig"].encode("utf-8") + "\n")
        f_view.write("cs:\n")
        for v in s["cs_alternatives"]:
            f_view.write(v.encode("utf-8") + "\n")
        f_view.write("en:\n")
        for v in s["en_alternatives"]:
            f_view.write(v.encode("utf-8") + "\n")
        f_view.write("sp:\n")
        for v in s["sp_alternatives"]:
            f_view.write(v.encode("utf-8") + "\n")
