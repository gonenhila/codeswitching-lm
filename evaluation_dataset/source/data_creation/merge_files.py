
import json
import argparse
import pdb
import random

random.seed(10)

parser = argparse.ArgumentParser()
parser.add_argument("--fname1", help="name of input file1")
parser.add_argument("--fname2", help="name of input file2")
parser.add_argument("--output", help="name of output file")
args = parser.parse_args()

with open("../../data/alternatives/alternate_sents_{}.json".format(args.fname1), "r") as f1, open("../../data/alternatives/alternate_sents_{}.json".format(args.fname2), "r") as f2:
    all_sets1 = json.load(f1)
    all_sets2 = json.load(f2)

all_sets = all_sets1 + all_sets2

random.shuffle(all_sets)

with open("../../data/alternatives/alternate_sents_{}.json".format(args.output), "w") as f:
    json.dump(all_sets, f)
