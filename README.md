# OAR-soccer
OARNet for soccer event predicting

### Requirement

```
python>=3.7
tqdm>=4.63.0
pytorch>=1.10
dgl==0.8.0
```

### Data

#### soccER dataset
Prepare data by downloading file "(train/test)\_(event-type)\_delay_(delay-type)\_mode_(edge-type).hdf5" to "data" directory.

For example, "train_pass_delay_0_mode_0.hdf5" is a training dataset whose edge mode is player-centered, event type is pass and delay type is 0.

download link: 
https://drive.google.com/drive/folders/1H_E1stiQISVt4Lb04eXAucXoqPdhK6Hx?usp=sharing

#### skillcorner dataset
Prepare data by downloading files to data/matches directory.

download link: 
https://drive.google.com/drive/folders/1zmRxW2xHjGcWQkQ6gqqjtpvTAPnnOfZF?usp=sharing

### Arguments
--attmode:
&emsp; 0 $\rightarrow$ GAT-style attention
&emsp; 1 $\rightarrow$ average attention
&emsp; 2 $\rightarrow$ random attention
&emsp; 3 $\rightarrow$ multihead GRU attention

--edgemode:
&emsp; 0 $\rightarrow$ player-centered
&emsp; 1 $\rightarrow$ ball-centered
&emsp; 2 $\rightarrow$ fixed

--multiatt:
&emsp; True $\rightarrow$ multihead attention
&emsp; False $\rightarrow$ onehead attention

--gnn:
&emsp; gat
&emsp; sage
&emsp; gin
&emsp; ggnn

--delay:
&emsp; 0 $\rightarrow$ event recognition
&emsp; 1 $\rightarrow$ event prediction
&emsp; 2 $\rightarrow$ intention prediction

### Train and valid
run ``` python train_pass_soccer.py ``` to perform training process for pass event prediction on soccER dataset.
run ``` python train_shot_soccer.py ``` to perform training process for shot event prediction on soccER dataset.
run ``` python train_pass_sc.py ``` to perform training process for pass event prediction on skillcorner dataset.
run ``` python train_shot_sc.py ``` to perform training process for shot event prediction on skillcorner dataset.