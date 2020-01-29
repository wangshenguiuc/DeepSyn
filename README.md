# DeepSyn
> Deep functional synthesis: a machine learning approach to gene function enrichment

## Introduction
Deep Functional Synthesis (DeepSyn) is function annotation tool that moves beyond gene function databases to dynamically infer the functions of a gene set from its associated network of literature and data, conditioned on the disease and drug context of the current experiment.

## Publication

**Deep functional synthesis: a machine learning approach to gene function enrichment**.  
*under review*.
*Sheng Wang, Jianzhu Ma, Samson Fong, Stefano Rensi, Jiawei Han, Jian Peng, Dexter Pratt, Russ B. Altman, Trey Ideker*.

**Annotating gene sets by mining large literature collections with protein networks**.
*PSB 2018*.
*Wang, Sheng, Jianzhu Ma, Michael Ku Yu, Fan Zheng, Edward W. Huang, Jiawei Han, Jian Peng, Trey Ideker*

## Dataset (update on 01/27/2020)
We now provide an updated dataset for our fast implementation in this [dropbox link](https://www.dropbox.com/s/ubkh36hqe8qqki0/deepsyn_data.tar.gz?dl=0).
Please download this data, extract all files (tar -zxvf deepsyn_data.tar.gz) and change the data fold location in [run_deepsyn_fast.py](https://github.com/wangshenguiuc/DeepSyn/blob/master/run_deepsyn_fast.py).

## How to run
DeepSyn supports queried in four different fields: drug, gene, function, and disease.

Users can give arbitary number of terms in any of these four fields.
DeepSyn will then return an answer graph for this query.

First modify the [run_deepsyn_fast.py](https://github.com/wangshenguiuc/DeepSyn/blob/master/run_deepsyn_fast.py) and change DATA_DIR to where you store the dataset downloaded from [dropbox link](https://www.dropbox.com/s/ubkh36hqe8qqki0/deepsyn_data.tar.gz?dl=0).
An example is in run_deepsyn_fast.py
```
python run_deepsyn_fast.py
```

## Prerequisite
* python 3.6 (with slight modification, python 2.7 can also be used to run our tool)
* python packages (numpy 1.14+, scipy 1.1+)

## Questions
For questions about the data and code, please contact swang91@stanford.edu. We will do our best to provide support and address any issues. We appreciate your feedback!
