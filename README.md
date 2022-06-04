<h2 align="center">Recommender System WITH PyTorch 🟢🟠🔴 </h2>

PyTorch Implementation of classic Recommender System Models mainly used for **self-learing&communication**.

> checkout for [tensorflow](https://github.com/nbsps/RS-Models/tree/tensorflow) branch

> **corresponding papers** 🚀 [RS_Papers 📖](https://github.com/nbsps/RS_CA_Papers)

## Matching

### Matrix Factorization

| Model       | dataset   | loss_func | metrics       | state |
| ----------- | --------- | --------- | ------------- | ----- |
| **LFM**     | `ml-100k` | MSELoss   | `MSE: 0.9031` | 🟢     |
| **BiasSVD** | `ml-100k` | MSELoss   | `MSE: 0.8605` | 🟢     |
| **SVD++**   | `ml-100k` | MSELoss   | `MSE: 0.8493` | 🟢     |

### Factorization Machine

| Model   | dataset  | loss_func | metrics       | state |
| ------- | -------- | --------- | ------------- | ----- |
| **FM**  | `criteo` | BCELoss   | `AUC: 0.6934` | 🟢     |
| **FFM** | `criteo` | BCELoss   | `AUC: 0.6729` | 🟢     |

### Sequential based

| Model      | dataset   | loss_func         | metrics                            | state |
| ---------- | --------- | ----------------- | ---------------------------------- | ----- |
| **FPMC**   | `ml-100k` | sBPRLoss          | `Recall@10: 0.0622`                | 🟢     |
| **SASRec** | `ml-100k` | BCEWithLogitsLoss | `NDCG@10: 0.1801` ` HR@10: 0.3595` | 🟢     |

### Knowledge aware

| Model         | dataset | loss_func            | metrics       | state |
| ------------- | ------- | -------------------- | ------------- | ----- |
| **RippleNet** | `ml-1m` | BCELoss              | `AUC: 0.8838` | 🟢     |

### Graph embedding

| DeepWalk | Node2vec | EGES |
| -------- | -------- | ---- |

### Point of Interests

| MIND | SDM  |
| ---- | ---- |

### CF

| Model        | dataset   | loss_func | metrics       | state |
| ------------ | --------- | --------- | ------------- | ----- |
| **NeuralCF** | `ml-100k` | MSELoss   | `MSE: 0.3322` | 🟢     |

## Ranking

### FM

| Model      | dataset  | loss_func | metrics       | state |
| ---------- | -------- | --------- | ------------- | ----- |
| **FNN**    | `criteo` | BCELoss   | `AUC: 0.6787` | 🟢     |
| **DeepFM** | `criteo` | BCELoss   | `AUC: 0.6854` | 🟢     |
| **NFM**    | `criteo` | BCELoss   | `AUC: 0.6705` | 🟢     |
| **AFM**    | `criteo` | BCELoss   | `AUC: 0.6572` | 🟢     |

### LR

| GBDT+LR |
| ------- |

### DNN

| Model             | dataset       | loss_func | metrics       | state |
| ----------------- | ------------- | --------- | ------------- | ----- |
| **Deep Crossing** | `criteo`      | BCELoss   | `AUC: 0.7210` | 🟢     |
| **PNN**           | `criteo`      | BCELoss   | `AUC: 0.6360` | 🟢     |
| **Wide&Deep**     | `criteo`      | BCELoss   | `AUC: 0.7074` | 🟢     |
| **DCN**           | `criteo`      | BCELoss   | `AUC: 0.7335` | 🟢     |
| **DIN**           | `amazon book` | BCELoss   | `AUC: 0.5988` | 🟢     |

> DIN: It seems that the feature engineering(negative sampling) of paper used for `amazon book` seems bad. I try hard but the auc of test cannot reach the `0.811` on `amazon book`.

### Multi tasks

| Model    | dataset       | loss_func         | metrics                                       | state |
| -------- | ------------- | ----------------- | --------------------------------------------- | ----- |
| **MMOE** | census-income | BCEWithLogitsLoss | `income-AUC: 0.9061` `marry-AUC: 0.9637`      | 🟢     |
| **ESMM** | census-income | BCEWithLogitsLoss | `income-ctr-AUC:  0.9242` `ctcvr-AUC: 0.9122` | 🟢     |



