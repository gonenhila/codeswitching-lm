# Language Modeling for Code-Switching

This project includes the code for the models described in the [paper](https://arxiv.org/pdf/1810.11895.pdf): 

**"Language Modeling for Code-Switching: Evaluation, Integration of Monolingual Data, and Discriminative Training"**, Hila Gonen and Yoav Goldberg, arXiv:1810.11895

## Prerequisites

* Python 2.7
* DyNet 2.0

## Fetch Data

If you haven't done so yet, as a first step after cloning/downloading the repository, you should fetch all data files using the **fetch_data.sh** script as follows (from the **codeswitching** folder):
```
sh fetche_data.sh
```
This will download all data files needed for this repository into the respective directories.

## Important notes

* Using a **GPU and autobatching** when training a model is strongly recommended, this is done with the following flags:
```
--dynet-gpus 1 --dynet-autobatch 1
```
* Sometimes the default **memory allocation** is not enough, and more memory should be allocated:
```
--dynet-mem <mem-size> 
```
* **Naming convention:** when providing a name X for 'logfile', a log file (X.txt) and a model (X_model) are saved during the run. The default name is 'log'.

## Language Model - Standard training

Training standard language models is done from the **generative/source/** folder:


```
cd generative/source/
```



### Training a standard language model

To train a standard language model:
```
python train_lm.py --dynet-weight-decay 0.00001 --new_test [--logfile <name>] 
```


### Training a fine-tuned standard language model

To train a model using the fine-tune protocal, first run the first phase - pretraining a model using monolingual data:
```
python train_lm.py --dynet-weight-decay 0.00001 --new_test --check_freq 6000 --finetune_p1 --logfile <name>
```
where **name** is the name of the model to be saved at the end of the pretraining phase.

Then, to fine-tune the model using the CS data:
```
python train_lm.py --dynet-weight-decay 0.00001 --new_test --finetune_p2 <name_of_pretrained_model> [--logfile <name>]
```
where **name_of_pretrained_model** is **name** from the first phase.

### Loading a pretrained model

To load and evaluate a pretrained model (the model should be saved under the **generative/models/** folder, with '_model' ending):
```
python train_lm.py --dynet-weight-decay 0.00001 --new_test --load_model <model_to_load> [--logfile <name>]

```

A pretrained fine-tuned language model can be downloaded from [here](http://u.cs.biu.ac.il/~gonenhi/models/lm_cs/generative/finetune_model). 

You can load and evaluate it (after saving it under the **generative/models/** folder.) as follows:
```
python train_lm.py --dynet-weight-decay 0.00001 --new_test --load_model finetune [--logfile <name>]
```

## Discriminative Training

Training a model using discriminative training is done from the **discriminative/source/** folder:


```
cd discriminative/source/
```



### Training a model using discriminative training

To train a model using discriminative training:
```
python train_lm_sets.py --logfile <name>
```


### Training a discriminative fine-tuned model

To train a discriminative model using the fine-tune protocal, first run the first phase - pretraining a model using monolingual data:
```
python train_lm_sets.py --check_freq 6000 --finetune_p1 --logfile <name>
```
where **name** is the name of the model to be saved at the end of the pretraining phase.

Then, to fine-tune the model using the CS data:
```
python train_lm_sets.py --finetune_p2 <name_of_pretrained_model> [--logfile <name>]
```
where **name_of_pretrained_model** is **name** from the first phase.


### Loading a pretrained model

To load and evaluate a pretrained model (the model should be saved under the **generative/models/** folder, with '_model' ending):
```
python train_lm_sets.py --load_model <model_to_load> [--logfile <name>]

```

A pretrained fine-tuned discriminative model can be downloaded from [here](http://u.cs.biu.ac.il/~gonenhi/models/lm_cs/discriminative/finetune_model). 

You can load and evaluate it (after saving it under the **discriminative/models/** folder.) as follows:
```
python train_lm_sets.py --load_model finetune [--logfile <name>]
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


