# DeepSyn
> Deep functional synthesis: a machine learning approach to gene function enrichment

## Introduction
Deep Functional Synthesis (DeepSyn) is function annotation tool that moves beyond gene function databases to dynamically infer the functions of a gene set from its associated network of literature and data, conditioned on the disease and drug context of the current experiment.

## Publication (under review)

**Deep functional synthesis: a machine learning approach to gene function enrichment**.
*Sheng Wang, Jianzhu Ma, Samson Fong, Stefano Rensi, Jiawei Han, Jian Peng, Dexter Pratt, Russ B. Altman, Trey Ideker*.

## Dataset
We provide a sample dataset in [figshare](https://figshare.com/projects/DeepSyn_Deep_functional_synthesis_a_machine_learning_approach_to_gene_function_enrichment/69380), including
(1) the knowledge graph used in the paper
(2) all the raw data used to generate the knowledge graph except for the scientific paper abstracts (i.e., PubMed)
(3) new phrases mined by using AutoPhrase
(4) co-occured phrase networks mined from PubMed (so you don't need to download PubMed raw data)
(5) CNN-based phrase networks mined from PubMed

By using (1), it would be enough for all functions in our tool.
If you want to create your own knowledge graph or replicate our results, you can use (2), (3), (4), and (5) to do it.

## How to run
DeepSyn supports queried in four different fields: drug, gene, function, and disease.
Users can give arbitary number of terms in any of these four fields.
DeepSyn will then return an answer graph for this query.

First modify the src/configure.py and change DATA_DIR to where you store the dataset downloaded from [figshare](https://figshare.com/projects/DeepSyn_Deep_functional_synthesis_a_machine_learning_approach_to_gene_function_enrichment/69380).
An example is in src/run_DeepSyn.py
```
cd src
python run_DeepSyn.py
```
The resulted answer graph would be stored in network.txt in the format that can be read by [CytoScape](https://cytoscape.org/). If [graphviz](https://pypi.org/project/graphviz/) is installed, a graph will be generated in network.pdf

## Prerequisite
python 3.6 (with slight modification, python 2.7 can also be used to run our tool)
python packages (numpy 1.14+, scipy 1.1+, networkx 2.3+)

## Questions
For questions about the data and code, please contact swang91@stanford.edu. We will do our best to provide support and address any issues. We appreciate your feedback!
