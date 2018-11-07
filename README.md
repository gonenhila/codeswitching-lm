# Language Modeling for Code-Switching

This project includes the data and models described in the [paper](https://arxiv.org/pdf/1810.11895.pdf): 

**"Language Modeling for Code-Switching: Evaluation, Integration of Monolingual Data, and Discriminative Training"**, Hila Gonen and Yoav Goldberg, arXiv:1810.11895 

## Prerequisites

* Python 2.7
* DyNet 2.0
* [Carmel Finite-State Toolkit](https://www.isi.edu/licensed-sw/carmel/)

## Fetch Data

As a first step, after cloning/downloading the repository, fetch all data files using the **fetch_data.sh** script as follows (from the **codeswitching** folder):
```
sh fetche_data.sh
```
This will download all data files needed for this repository into the respective directories.

## Dataset

Please refer to the detailed [README](evaluation_dataset/README.md) in the **evaluation_dataset** folder.

## Models

Please refer to the detailed [README](language_model/README.md) in the **language_model** folder.

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

This project is licensed under Apache License - see the [LICENSE](LICENSE) file for details.


