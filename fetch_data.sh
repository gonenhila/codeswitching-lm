#!/bin/bash
cd evaluation_dataset/data/
wget http://u.cs.biu.ac.il/~gonenhi/code/evaluation_dataset/data/alternatives.tar.gz
wget http://u.cs.biu.ac.il/~gonenhi/code/evaluation_dataset/data/FSTs.tar.gz
tar -zxvf alternatives.tar.gz alternatives/
tar -zxvf FSTs.tar.gz FSTs/
cd ../../language_model/
wget http://u.cs.biu.ac.il/~gonenhi/code/language_model/data.tar.gz
tar -zxvf data.tar.gz data/
