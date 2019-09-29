# DeepSyn
> Deep functional synthesis: a machine learning approach to gene function enrichment

## Introduction
Deep Functional Synthesis (DeepSyn) is function annotation tool that moves beyond gene function databases to dynamically infer the functions of a gene set from its associated network of literature and data, conditioned on the disease and drug context of the current experiment.

## Publication (under review)

**Deep functional synthesis: a machine learning approach to gene function enrichment**.
*Sheng Wang, Jianzhu Ma, Samson Fong, Stefano Rensi, Jiawei Han, Jian Peng, Dexter Pratt, Russ B. Altman, Trey Ideker*.

## How to run
DeepSyn supports queried in four different fields: drug, gene, function, and disease.
Users can give arbitary number of terms in any of these four fields.
DeepSyn will then return an answer graph for this query.

A detailed example is in run_server.py