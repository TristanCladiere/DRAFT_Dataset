# DRAFT: A Dataset for Recruitment Assessment and Fairness Tracking

## Authors

Tristan Cladière<sup>1</sup>, Antoine Gourru<sup>1</sup>, Bissan Audeh<sup>2</sup>, Christine Largeron<sup>1</sup>

- <sup>1</sup> Laboratoire Hubert Curien, UMR CNRS 5516, 42000 Saint-Etienne, France

- <sup>2</sup> Konduco, 69140 Rillieux-la-Pape, France

## Acknowledgments
This work was partially supported by the **French Direction de l'Animation de la Recherche, des Études et des Statistiques** (DARES) 
through the IA et recrutement project, by the **EABCD Ambition internationale project** (Région Auvergne-Rhône-Alpes), 
and by the **French ANR project FAMOUS** (ANR-23-CE23-0019).

## Citation
T. Cladière, A. Gourru, B. Audeh, C. Largeron. DRAFT: A Dataset for Recruitment Assessment and Fairness Tracking. AI4HR&PES @ ECML-PKDD (2026).


## Installation
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-31312/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/previous-versions/)

(Tested on Windows 10/11)
### Requirements
- Python 3.13
- PyTorch 2.7.1
- CUDA 12.8

You can install torch and cuda with:
```
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

Then install other dependencies: 
```
pip install -r requirements.txt
```

### Download data
You can download the data from [here](https://huggingface.co/datasets/LabHC/DRAFT).
Then, place the "Data" folder inside the project, or specify your path using `--data_folder your/path/to/Data` when 
running `main.py`, `test_embeddings.py`, and `expected_scores.py`.


## Training/Testing FC layer
To train the FC layer, run:
```
python main.py --k_cross 1-5
```
Here, '1' is the test fold, and '5' is the number of folds for K-folds cross-validation. With K=5, you must therefore 
run this script 5 times, and change the test fold each time (for example: `--k_cross 2-5`). When all the trainings are 
completed, you can summarize the results using:

```
python summarize_k_cross_results.py --expe_folder output/0.6B/ES-500_LR-0.0005/5_cross_val 
```
Note: you must change the path depending on your training configuration (see args in `main.py`)

## Cosine similarity/Leakage
To evaluate the cosine similarity between jobs and candidates embeddings, as well as gender leakage in candidates 
embeddings, you can run:
```
python test_embeddings.py 
```

## Expected scores/Data bias
To evaluate the best possible scores, the bias in the data, and the expected scores under random predictions, you can 
run:
```
python expected_scores.py 
```
