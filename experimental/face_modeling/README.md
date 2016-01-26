# Head Modeling
---------
# Part 1: Detecting Head Turns
Collect, label and train when head turns occur.

## Collect
Run:
```
python collect.py <event_type> 
```

## Label
Run:
```
python cluster_merge.py <data_dir> <# of clusters>
```

## Train
Run:
```
python train.py head_turns
```

