# AnimalNet

A deep neural net that classifies images from a dataset as animals or not.

## Getting Started
```
git clone https://github.com/sgaur98/AnimalNet.git
cd AnimalNet/
```

### Prerequisites
1) Install python3
2) Install PyTorch
3) Install NumPy
4) Unzip mp6_data.zip

### Execution
```
usage: mp6.py [-h] [--dataset DATASET_FILE] [--max_iter MAX_ITER]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET_FILE
                        the directory of the training data
  --max_iter MAX_ITER   Maximum iterations - default 10

```
#### Example
```
$ python3 mp6.py --dataset mp6_data
```
Output:
```
Accuracy: 0.8456
F1-Score: 0.8693297224102912
Precision: 0.8669817690749494
Recall: 0.8716904276985743
```
  
 
