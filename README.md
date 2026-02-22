# CrossAffinity: Protein-Protein Binding Affinity Prediction

CrossAffinity is a protein-protein binding affinity prediction model which separates the protein complex into two parts. The input to the model are the sequences belonging to both parts of the protein complex.

---
<img width="1155" height="434" alt="Architecture" src="https://github.com/user-attachments/assets/2d5596c0-66f3-419c-81b8-4519ef7da69d" />

## Table of Contents

- [Background](#background)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Testing](#test)
- [References](#references)

---

## Background

- **Problem:** Prediction of peptide and protein binders useful in many context such as identifying antibody sequences or peptide drugs with high binding affinity to the protein of interest.
- **Motivation:** The structure of protein complexes is often unavailable for the many published structure-based binding affinity models. CrossAffinity is useful in such cases, such as filtering of top-performing peptide drugs, where it is impossible to elucidate the structure of every single peptide candidate.
- **Task definition:**  
  - Input: CSV file with the protein sequence of the complex. Each sequence in each part is separated by ";". Example of part 1 sequence is "HKLA;GGYT" and part 2 sequence is "LDKD". The format of csv input file can be found in the example folder.
  - Output: Predicted pKd of the complex between part 1 and part 2 of the protein complex. An example of an output file can be found in the example folder.

---

## Features

- [✓] Predicts [pKd] for protein–protein pairs.
- [✓] Supports [sequence-based] protein representations.
- [✓] Trained on [PDBbind] subset dataset, from training set of ProAffinity-GNN by Zhou et al.
- [✓] Provides ready-to-use inference script and batching on GPUs.

---

## Installation
Create a suitable Python environment to train a new model or infer the pre-trained CrossAffinity model.

```bash
conda create -n CrossAffinity python=3.11
conda activate CrossAffinity

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install torch-geometric
pip install fair-esm
pip install scikit-learn pandas scipy openpyxl
pip install mlflow==3.2.0

git clone https://github.com/CodeJS19/CrossAffinity.git
```

---

## Usage
Select the input file, number of workers, output file path and devices for ESM2 and CrossAffinity models. More options can be accessed by reading `python inference.py -h`.

```bash
cd CrossAffinity
python inference.py --filepath example/example_input.csv --output example/example_output.csv --esm2_device cpu --num_workers 2
```

---

## Training
CrossAffinity can be retrained using the prepare.py, which prepares the training set, and Train.py, which trains the model with 5-fold cross validation. More options can be accessed by reading `python prepare.py -h` and `python Train.py -h`. Training can be monitored at localhost:8080.

```bash
python prepare.py --esm2_device cpu --num_workers 2
python Train.py

mlflow server --host 127.0.0.1 --port 8080
```

---

## Test
Please obtain the test set from the ProAffinity-GNN and the PPAP study by Zhou et al. and Qian et al., respectively.

---

## References

Please cite the following work if the test sets are obtained from their repository:

Zhou, Z.; Yin, Y.; Han, H.; Jia, Y.; Koh, J. H.; Kong, A. W.-K.; Mu, Y. ProAffinity-GNN: a novel approach to structure-based protein-Protein binding affinity prediction via a curated data set and graph neural networks. Journal of chemical information and modeling 2024, 64, 8796-8808.

Qian, J.; Yang, L.; Duan, Z.; Wang, R.; Qi, Y. PPAP: A Protein-protein Affinity Predictor Incorporating Interfacial Contact-Aware Attention. Journal of Chemical Information and Modeling 2025, 65, 9987-9998.
