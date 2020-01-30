# DeepSyn
> Deep functional synthesis: a machine learning approach to gene function enrichment

## Introduction
Deep Functional Synthesis (DeepSyn) is function annotation tool that moves beyond gene function databases to dynamically infer the functions of a gene set from its associated network of literature and data, conditioned on the disease and drug context of the current experiment.

## Publication
**Deep functional synthesis: a machine learning approach to gene function enrichment**.  
*under review*  
*Sheng Wang, Jianzhu Ma, Samson Fong, Stefano Rensi, Jiawei Han, Jian Peng, Dexter Pratt, Russ B. Altman, Trey Ideker*.

**Annotating gene sets by mining large literature collections with protein networks**.  
*PSB 2018*  
*sheng Wang, Jianzhu Ma, Michael Ku Yu, Fan Zheng, Edward W. Huang, Jiawei Han, Jian Peng, Trey Ideker*.

## Dataset (update on 01/27/2020)
We now provide an updated dataset for our fast implementation in this [dropbox link](https://www.dropbox.com/s/ubkh36hqe8qqki0/deepsyn_data.tar.gz?dl=0).
Please download this data, extract all files (tar -zxvf deepsyn_data.tar.gz) and change the data fold location in [run_deepsyn_fast.py](https://github.com/wangshenguiuc/DeepSyn/blob/master/run_deepsyn_fast.py).

## How to run
DeepSyn supports queried in four different fields: drug, gene, function, and disease.  
Users can give arbitary number of terms in any of these four fields.
DeepSyn will then return an answer graph for this query.  
An example is in run_deepsyn_fast.py.  
First download the dataset from [dropbox link](https://www.dropbox.com/s/ubkh36hqe8qqki0/deepsyn_data.tar.gz?dl=0).  
The following command runs the query for drug: [etoposide], disease: [breast cancer, lung cancer], function: [dna repair], gene: [top2a; kmt2a]. data_dir will be the folder that you store the dataset downloaded from [dropbox link](https://www.dropbox.com/s/ubkh36hqe8qqki0/deepsyn_data.tar.gz?dl=0).
```
python run_deepsyn_fast.py \
--data_dir "DeepSyn/data/knowledge_network/" \
--drug "doxorubicin" --disease "breast cancer;lung cancer" \
--gene "top2a;kmt2a" --function "dna repair" --output_file "test.txt"
```

Try the following command to see all the arguments.
```
python run_deepsyn_fast.py --help
```
## How to see output
The output graph will be stored in a networkx object nx_obj in run_deepsyn_fast.py.  
The graph will also be saved in text file by using the --output_file argument. The output file can be directly read by Cytoscape to generate a graph.


## Prerequisite
* python 3.6 (with slight modification, python 2.7 can also be used to run our tool)
* python packages (numpy 1.14+, scipy 1.1+, networkx 2.4+)

## Questions
For questions about the data and code, please contact swang91@stanford.edu. We will do our best to provide support and address any issues. We appreciate your feedback!
