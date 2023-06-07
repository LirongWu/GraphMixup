# GraphMixup


This is a PyTorch implementation of the GraphMixup, and the code includes the following modules:

* Dataset Loader (Cora, BlagCatalog, and Wiki-CS)

* Various Architectures (GCN, SAGE, GAT, and SEM)

* Five compared baselines (Origin, Over-Sampling, Re-weight, SMOTE, and Embed-SMOTE)

* Training paradigm (joint learning, pre-training, and fine-tuning) for node classification on three datasets

* Visualization and evaluation metrics 

  

## Main Requirements

* networkx==2.5
* numpy==1.19.2
* scikit-learn==0.24.1
* scipy==1.5.2
* torch==1.6.0



## Description

* train.py  
  * train() -- Train a new model for node classification task on the *Cora, BlagCatalog, and Wiki-CS* datasets
  * test() -- Test the learned model for node classification task on the *Cora, BlagCatalog, and Wiki-CS* datasets
  * save_model() -- Save the pre-trained model
  * load_model() -- Load model for fine-tuning
* data_load.py  
  
  * load_cora() -- Load Cora Dataset
  * load_BlogCatalog() -- Load BlogCatalog Dataset
  * load_wiki_cs() -- Load Wiki-CS Dataset
* models.py  
  
  * GraphConvolution() -- GCN Layer
  * SageConv() -- SAGE Layer
  * SemanticLayer() -- Semantic Feature Layer
  * GraphAttentionLayer() -- GAT Layer
  * PairwiseDistance() -- Perform self-supervised Local-Path Prediction
  * DistanceCluster() -- Perform self-supervised Global-Path Prediction
* utils.py  
  * src_upsample() -- Perform interpolation in the input space
  * src_smote() -- Perform interpolation in the embedding space
  * mixup() -- Perform mixup in the semantic relation space
* QLearning.py  
  * GNN_env() -- Calculate rewards, perform actions, and update states
  * isTerminal() -- Determine whether the termination conditions have been met



## Running the code

1. Install the required dependency packages

3. To get the results on a specific *dataset*, first run with proper hyperparameters to perform pre-training

  ```
python train.py --dataset data_name --setting pre-train
  ```

where the *data_name* is one of the 3 datasets (CCora, BlagCatalog, and Wiki-CS). The pre-trained model will be saved to the corresponding checkpoint folder in **./checkpoint** for evaluation.

3. To fine-tune the pre-trained model, run

  ```
python train.py --dataset data_name --setting fine-tune --load model_path
  ```

where the *model_path* is the path where the pre-trained model is saved.

4. We provide five compared baselines in this code. They can be configured via the '--setting' arguments:

- Origin: Vanilla backbone models with *'--setting raw'*
- Over-Sampling: Repeat nodes in the minority classes with *'--setting over-sampling'*
- Re-weight: Give samples from minority classes a larger weight when calculating the loss with *'--setting re-weight'*
- SMOTE: Interpolation in the input space with *'--setting smote'*
- Embed-SMOTE: Perform SMOTE in the intermediate embedding space with *'--setting embed_smote'*

Use *Embed-SMOTE* as an example: 

  ```
python train.py --dataset cora --setting embed_smote
  ```


## Citation

If you find this project useful for your research, please use the following BibTeX entry.

<<<<<<< HEAD
## Citation

If you find this file useful in your research, please consider citing:
=======
>>>>>>> 97cf8df6c2fcd7b5e029f7406aade36a408e8b90

```
@inproceedings{wu2023graphmixup,
  title={Graphmixup: Improving class-imbalanced node classification by reinforcement mixup and self-supervised context prediction},
  author={Wu, Lirong and Xia, Jun and Gao, Zhangyang and Lin, Haitao and Tan, Cheng and Li, Stan Z},
  booktitle={Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2022, Grenoble, France, September 19--23, 2022, Proceedings, Part IV},
  pages={519--535},
  year={2023},
  organization={Springer}
}
```
