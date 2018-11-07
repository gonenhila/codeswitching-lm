# Language Modeling for Code-Switching

This project includes the dataset described in the [paper](https://arxiv.org/pdf/1810.11895.pdf): 

**"Language Modeling for Code-Switching: Evaluation, Integration of Monolingual Data, and Discriminative Training"**, Hila Gonen and Yoav Goldberg, arXiv:1810.11895

## Prerequisites

* Python 2.7
* [Carmel Finite-State Toolkit](https://www.isi.edu/licensed-sw/carmel/)
(make sure that the 'carmel' executable file is in your '$PATH' variable, so that using the command 'carmel' in the command line executes this program)

## Fetch Data

If you haven't done so yet, as a first step after cloning/downloading the repository, you should fetch all data files using the **fetch_data.sh** script as follows (from the **codeswitching** folder):
```
sh fetche_data.sh
```
This will download all data files needed for this repository into the respective directories.

## Evaluation dataset

The evaluation dataset can be found under the **data/alternatives/** folder: [dev](data/alternate_sents_dev_filtered.json) and [test](data/alternate_sents_test_filtered.json)

## Creating data using FSTs

All the code needed for creating sets of alternative sentences can be found under the **source/** folder.

### for English-Spanish code-switched text

In order to create alternatives for English-Spanish code-switched sentences:
```
cd source/data_creation/
python create_alternatives.py --input_file <input_file> --output_file <output_file> --pool <num_of_threads>
```
The input file should have language tags in the following format:

* for English words: word__en
* for Spanish words: word__sp

You can later filter the created alternatives (if used for evaluation, use the **--eval** flag to limit the number of sets):

```
python filter_alternatives.py --fname <input_file>
```


### Creating data for other languages

In order to create alternative sentences for other langauges, say l1 and l2, the following is required:

* Language ID tags per word (each word should end with "\_\_l1" or "\_\_l2", where "l1" and "l2" are the chosen tags for the languages).
* Compatible pronunciation dictionaries for both l1 and l2 (saved under **data/dictionaries/**, same format as [dict_en](data/dictionaries/dict_en), the [English CMU dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict))
* Unigram word probabilities (saved under **data/probs/**, same format as [probs_en](data/probs/probs_en.txt))

#### Creating FSTs

```
cd source/FST_creation/
python create_lang_fsts.py --l1 <l1> --l2 <l2> --probs_l1 <probs_l1> --probs_l2 <probs_l2> --dict_l1 <dict_l1> --dict_l2 <dict_l2>
```
If you want to allow changes in the phoneme sequence before decoding into sentences, you can also create a third FST:
```
python create_fst_change_phones.py --mapping <path to mapping>
```
The mapping should be in the same format as [this one](data/mappings/mappings_subs.json).


#### Dictionaries

Should be compatible to each other, in the same format as the [English CMU dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict), that is found also in the **data/dictionaries/** folder: [dict_en](data/dictionaries/dict_en).


#### Creating the data

```
cd source/data_creation/
python create_alternatives.py --input_file <input_file> --output_file <output_file> --pool <num_of_threads> --l1 <l1> --l2 <l2>
```
Note you should also add the paths to the probability files and to your created FSTs.

You can later filter the created alternatives (if used for evaluation, use the **--eval** flag to limit the number of sets.):

```
python filter_alternatives.py --fname <input_file>
```


## Cite

If you find this project useful, please cite the paper:
```
@article{gonen2018,
  title   ={Language Modeling for Code-Switching: Evaluation, Integration of Monolingual Data, and Discriminative Training},
  author  ={Gonen, Hila and Goldberg, Yoav},
  journal ={arXiv preprint arXiv:1810.11895},
  year    ={2018}
}
```

## Contact

If you have any questions or suggestions, please contact [Hila Gonen](mailto:hilagnn@gmail.com).

## License

This project is licensed under Apache License - see the [LICENSE](../LICENSE) file for details.



