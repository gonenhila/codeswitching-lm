
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--emit_prob", default=0.0025, type=float, help="")
parser.add_argument("--substitute_prob", default=0.25, type=float, help="")
parser.add_argument("--prefer_sub", default=100, type=float, help="factor to prefer substitution over emission")
parser.add_argument("--mappings", default="../../data/mappings/mappings_subs.json", help="location of mappings file")
parser.add_argument("--output", default="../../data/FSTs/fst_change_phones", help="location of output file")
args = parser.parse_args()


with open(args.mappings, "r") as f:
    mappings = json.load(f)

# all_phones: list of all possible phones
# subs_mapping: mapping between phones and their permitted substitutions
subs_mapping, all_phones = mappings["subs_mapping"], mappings["all_phones"]

# build the fst that permits substitutions and emissions, according to the desired probabilities
f = open(args.output, "w")
f.write("%%% fst that maps phones to similar phones %%% \n 0 \n")
for k in all_phones:
    if k in subs_mapping:
        # do not change phone
        f.write('(0 (0 {} {} {}))\n'.format(k, k, 1-args.substitute_prob))
        # substitute to one of the permitted possibilities
        others = args.prefer_sub*len(subs_mapping[k]) + 1
        for v in subs_mapping[k]:
            f.write('(0 (0 {} {} {}))\n'.format(k, v, float(args.substitute_prob*args.prefer_sub)/others))
        # emit phone
        f.write('(0 (0 {} *e* {}))\n'.format(k, args.substitute_prob/others))
    else:
        # do not change phone
        f.write('(0 (0 {} {} {}))\n'.format(k, k, 1-args.emit_prob))
        # emit phone
        f.write('(0 (0 {} *e* {}))\n'.format(k, args.emit_prob))
