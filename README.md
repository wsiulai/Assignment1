# DE-HNN: An effective neural model for Circuit Netlist representation

## Installation
```
conda create -n python=3.9.18
conda activate
pip install -r requirements.txt
```

## Pre-trained Model (Folder "model")
The pre-trained DE-HNN model is trained on xbar dataset. We will maintain this general base DE-HNN model and further pre-train it on more unlabeled circuit samples given in the future.

 
## Netlist Data Collection (Folder: "data_collection")

* All datasets used are from the paper's github.
    


## Data Preprocessing (Folder: "data_collection")

* Raw data to processed data circuit to graph transformation
  
  - design2graph.py in folder "scr_design2graph"
  - Netlist transformation: net_design2graph.py in folder "netlist/scr"

    


## DE-HNN Pre-training (Folder: "model/dehnn")

### Pre-training 
```
cd ./model/dehnn
python3 pretrain_all.py
```
 
### DE-HNN model
- Folder "/model/dehnn/models"
- Change configurations in "/model/dehnn/configs"
- Model saved in "/model/dehnn/pretrain_model"






## Abstract

The optimization of power, performance, and area (PPA) are important core
aspects of chip design and hardware development in general. In addition,
chip design’s growing complexity has increased the time it takes to design a
working chip. In our domain, we investigate how graph neural networks as a
machine learning model can provide fast feedback to chip designers to maxi-
mize those core PPA goals. However, in order for the machine learning (ML)
model to be accurate, how to represent the design data is very important to
think about. This representation is usually an object call netlist that describes
cells and nets and how they are connected. In particular, a circuit design is
represented as a netlist composed of cells and nets, where cells refer to the
electronic units and nets refer to the connectivity (i.e. hyper-edges) among
the cells. The growing scale of circuits and hence the size of these netlist ob-
jects shows an opportunity to transform knowledge from old designs to new
ones
