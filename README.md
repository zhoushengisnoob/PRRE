#PRRE

This is a python implementation of my CIKM 2018 paper "PRRE: Personalized Relation Ranking Embedding for Attributed Networks", which considers the partial correlation between node attributes and network topology.

**This is the original version of PRRE, the advanced version will be released later**

## Requirements
* networkx
* numpy
* scipy
* scikit-learn

All required packages are defined in requirements.txt. To install all requirement, just use the following commands:
```
pip install -r requirements.txt
```

## Basic Usage

### Input Data 
Each dataset contains 3 files: edgelist, feature for training and label for evaluation.
```
1. dataset.edgelist: each line contains two connected nodes.
node_1 node_2
node_2 node_3
...

2. dataset.feature: this file has n lines.
feature_1 feature_2 ... feature_n
feature_1 feature_2 ... feature_n
...

3. dataset.label: each line represents a node and its class label.
node_1 label_1
node_2 label_2
...
```

### Run
To run PRRE, just execute the following command for node classification task and link prediction with different 'task' parameter:
```
python prre.py
```
For data visualization task, Embedding Projector is recommended to deal with the embedding result. 

### Cite
@inproceedings{zhou2018prre,
  title={PRRE: Personalized Relation Ranking Embedding for Attributed Networks},
  author={Zhou, Sheng and Yang, Hongxia and Wang, Xin and Bu, Jiajun and Ester, Martin and Yu, Pinggang and Zhang, Jianwei and Wang, Can},
  booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
  pages={823--832},
  year={2018},
  organization={ACM}
}
