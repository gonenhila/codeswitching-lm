

import string
import itertools
from operator import add
import argparse
import json
import codecs

parser = argparse.ArgumentParser()
parser.add_argument("--dict", default="../../data/dictionaries/es.dict", help="location of dictionary file for language 1")
parser.add_argument("--new_dict", default="../../data/dictionaries/dict_sp", help="location of dictionary file for language 2")
parser.add_argument("--mappings", default="../../data/mappings/dict_mappings.json", help="location of mappings file")
args = parser.parse_args()


def adapt_sp_dict(l2_l1_map):

    with codecs.open(args.dict, "r", "utf-8") as f_dict, codecs.open(args.new_dict, "w", "utf-8") as f_new_dict:
        for l in f_dict:
            k, v = l.strip().split(" ", 1)

            # Cartesian product between options so far, and all options for next phoneme
            v_temp = []
            v_new = " "
            for c in v.split():
                for elem in itertools.product(v_new, l2_l1_map[c]):
                    v_temp.append(elem[0] + " " + elem[1])
                v_new = v_temp
                v_temp = []
            v_new = [item.strip() for item in v_new]

            f_new_dict.write(k + "\t" + "\t".join(v_new) + "\n")

    return


if __name__ == '__main__':

    with open(args.mappings, "r") as f:
        l2_l1_map = json.load(f)

    adapt_sp_dict(l2_l1_map)

