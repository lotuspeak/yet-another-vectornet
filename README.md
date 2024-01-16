# Reimplement VectorNet :car:

Paper: [VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation](https://arxiv.org/abs/2005.04259)

Still under construction:

- [x] finish the feature preprocessor
- [x] finish the hierarchical GNN
- [x] overfit the tiny sample dataset
- [x] batchify the data and compute subgraph in parallel
- [X] evaluate results on DE / ADE metrics
- [x] refine the feature preprocessor (how to encode the features)
- [x] Check the correctness of hierarchical-GNN's implementation
- [x] run on the whole dataset (running)
- [x] add multi-GPU training (currently too slow, 2h an epoch)
- [ ] add uni-test for each modules
- [ ] More advanced trajectory predictor, generate diverse trajectories (MultiPath, or variational RNNs; current using MLP)
- [ ] add node feature completing module


Inplement a Vectornet: hierarchical GNN encoder (no feature completing) + MLP predictor, without node feature completing.

~~The performance on test is 3.255 on  minADE (K=1) v.s that in paper of 1.81.~~ (bug found in `GraphDataset`: the former implementation contained *self-loops connection* in graph data, which was wrong; and the preprocessed `dataset.pt` was also wrong; now the model is still trainning...)

After I fix the bug about self-loops in `Graph.Data`, I re-train the network with the same setting but only to find the performance on the validation set remains the same for about 2.6 of ADE, which was so disappointing. Notice that I only use the context (social + lanes) with about 5-10 meters around each agent (not enough machine for me), so I tried to change the context radius to 100 meters in `config.py` file (in the paper it's 200 * 200 if my memory serves me right). Unfortunately, the machines in the lab are not accessible to me right now, so I couldn't train the network with these new settings. :cry:

branch `master` is sync with branch `large-scale`; branch `overfit-small` is archived.


---

## Table of Contents

- [Environment](#Environment)
- [Usage](#Usage)
- [Results on val and test](#Results-on-val-and-test)
- [Result and visualization for overfitting tiny dataset](#Result-and-visualization-for-overfitting-tiny-dataset)

---

## Environment

Multi-GPU training on Windows Serer 2016; CUDA version 10.1; 2 Titan Xp GPUs.

Install the packages mentioned in requirements.txt
```
pip install -r requirements.txt
```

> torch==1.4.0, 
argoverse-api, 
numpy==1.18.1, 
pandas==1.0.0, 
matplotlib==3.1.1, 
torch-geometric==1.5.0

## Usage

For pre-processed data, pre-trained model, and results `*.h5` file: [Google Drive](https://drive.google.com/drive/folders/1XJ2Oz4Qc2UstnfRw3DNvQThuEVvM6tUL?usp=sharing)

(Remember to run `find . -name "*.DS_Store" -type f -delete` if you're using MacOS)

0) Install [Argoverse-api](https://github.com/argoai/argoverse-api/tree/master/argoverse). Download `HD-maps` in argoverse-api as instructed.

1) download [the prepared dataset objects on Google Drive](https://drive.google.com/drive/folders/1XJ2Oz4Qc2UstnfRw3DNvQThuEVvM6tUL?usp=sharing) directly and unzip it in path `.`, and skip step 3.

    or prepared the dataset (batchify ...) from raw *.csv. 
       
    put all data (folders named `train/val/test` or a single folder `sample`) in `data` folder.
    
    An example folder structure:
    ```
    data - train - *.csv
         \        \ ...
          \
           \- val - *.csv
            \       \ ...
             \
              \- test - *.csv
                       \ ...
    ```
2) Modify the config file `utils/config.py`. Use the proper env paths and arguments.

3) Feature preprocessing, save intermediate data input features (compute_feature_module.py)
    ```
    $ python compute_feature_module.py
    ```
    Use (200, 200) size for a single sequence as the paper told.

4) Train the model (`train.py`; overfit a tiny dataset by setting `small_dataset = True`, and use `GraphDataset` in `dataset.py` to batchify the data)
    ```
    $ python train.py
    ```

---

## Results on val and test

Some predicting results were uploaded to the Argoverse contest, check the board via the [url](https://evalai.cloudcv.org/web/challenges/challenge-page/454/leaderboard/)

Submission ID of the repo: @xkhuang

### Result on val


| model params                                                 | minADE (K=1) | minFDE (K=1) |
| ------------------------------------------------------------ | ------------ | ------------ |
| results in paper | 1.66  | 3.67  |
| epoch_24.valminade_2.637.200624.xkhuang.pth                  | 2.637        |              |

#### no rotated

    bs_1024_r1.log:minADE:3.518603, minFDE:7.254115, MissRate:0.929165
    bs_1024_r1.log:minADE:2.910138, minFDE:6.089996, MissRate:0.892759
    bs_1024_r1.log:minADE:2.477032, minFDE:5.303828, MissRate:0.801834
    bs_1024_r1.log:minADE:2.362716, minFDE:5.088229, MissRate:0.793119
    bs_1024_r1.log:minADE:2.277032, minFDE:4.929060, MissRate:0.762921
    bs_1024_r1.log:minADE:2.260361, minFDE:4.891617, MissRate:0.758614
    bs_1024_r1.log:minADE:2.258971, minFDE:4.896843, MissRate:0.761451
    bs_1024_r1.log:minADE:2.257784, minFDE:4.897358, MissRate:0.758994

    bs_256_r3_1.log:minADE:2.621358, minFDE:5.598257, MissRate:0.848906
    bs_256_r3_1.log:minADE:2.515848, minFDE:5.271922, MissRate:0.838924
    bs_256_r3_1.log:minADE:2.111074, minFDE:4.642370, MissRate:0.758816
    bs_256_r3_1.log:minADE:2.063809, minFDE:4.529336, MissRate:0.732950

    bs_256_r3_1.log:minADE:2.669239, minFDE:5.682845, MissRate:0.847411
    bs_256_r3_1.log:minADE:2.467033, minFDE:5.242740, MissRate:0.850527
    bs_256_r3_1.log:minADE:2.108477, minFDE:4.659002, MissRate:0.750887
    bs_256_r3_1.log:minADE:2.060997, minFDE:4.562815, MissRate:0.730721
    bs_256_r3_1.log:minADE:2.065531, minFDE:4.581269, MissRate:0.744730
    bs_256_r3_1.log:minADE:2.090754, minFDE:4.624962, MissRate:0.760539
    bs_256_r3_1.log:minADE:2.077855, minFDE:4.608405, MissRate:0.755472
    bs_256_r3_1.log:minADE:2.066780, minFDE:4.587068, MissRate:0.749240
    bs_256_r3_1.log:minADE:2.068860, minFDE:4.592795, MissRate:0.749899
    bs_256_r3_1.log:minADE:2.082567, minFDE:4.623325, MissRate:0.755472
    
    bs_256_r3.log:minADE:2.548336, minFDE:5.479505, MissRate:0.831678
    bs_256_r3.log:minADE:2.496875, minFDE:5.243385, MissRate:0.836390
    bs_256_r3.log:minADE:2.149730, minFDE:4.702978, MissRate:0.759120
    bs_256_r3.log:minADE:2.071554, minFDE:4.580588, MissRate:0.732798
    bs_256_r3.log:minADE:2.066520, minFDE:4.548301, MissRate:0.735737
    bs_256_r3.log:minADE:2.067908, minFDE:4.574391, MissRate:0.746960
    bs_256_r3.log:minADE:2.058419, minFDE:4.553373, MissRate:0.740145
    bs_256_r3.log:minADE:2.050795, minFDE:4.538275, MissRate:0.737535


#### rotated bs=256

    EXIST_THRESHOLD = (50)
    # index of the sorted velocity to look at, to call it as stationary
    STATIONARY_THRESHOLD = (13)
    color_dict = {"AGENT": "#d30000", "OTHERS": "#00d000", "AV": "#0000f2"}
    LANE_RADIUS = 30
    OBJ_RADIUS = 30

    minADE:2.111391, minFDE:4.654815, MissRate:0.753521 epoch 4
    minADE:2.011096, minFDE:4.493858, MissRate:0.737637 epoch 9
    minADE:1.842005, minFDE:4.112180, MissRate:0.691782 
    minADE:1.812565, minFDE:4.046578, MissRate:0.673718
    minADE:1.820584, minFDE:4.027604, MissRate:0.685093
    minADE:1.794450, minFDE:3.983025, MissRate:0.679393
    minADE:1.794970, minFDE:3.997268, MissRate:0.682864
    minADE:1.785888, minFDE:3.975405, MissRate:0.680761
    minADE:1.791395, minFDE:3.979677, MissRate:0.681166 epoch 44
    minADE:1.809420, minFDE:4.012702, MissRate:0.691123

#### 


### Result on test

| model params                                                 | minADE (K=1) | minFDE (K=1) |
| ------------------------------------------------------------ | ------------ | ------------ |
| results in paper | 1.81  | 4.01  |
| epoch_24.valminade_2.637.200624.xkhuang.pth                  | 3.255298     | 6.992046     |


---

## Result and visualization for overfitting tiny dataset

Sample results are shown below:
* red lines are agent input and ground truth output
* blue points are predicted feature tarjectory
* light blue lanes are other moving objects
* grey lines are lanes

### Using nearby context (about 5M around):
| | |
|:-------------------------:|:-------------------------:|
| ![](images/1.png) | ![](images/2.png) |
| ![](images/3.png) | ![](images/4.png) |

### Using 200 * 200 context (about 100M around):
with lanes:
| | |
|:-------------------------:|:-------------------------:|
| ![](images/200*200-1-1.png) | ![](images/200*200-2-1.png) |
| ![](images/200*200-3-1.png) | ![](images/200*200-4-1.png) |
| ![](images/200*200-5-1.png) |  |

without lanes:
| | |
|:-------------------------:|:-------------------------:|
| ![](images/200*200-1-2.png) | ![](images/200*200-2-2.png) |
| ![](images/200*200-3-2.png) | ![](images/200*200-4-2.png) |
| ![](images/200*200-5-2.png) |  |