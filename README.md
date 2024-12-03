# CircuitFusion: Multimodal Circuit Representation Learning for Agile Chip Design

## Installation
```
conda create -n ckt python=3.9.18
conda activate ckt
pip install -r requirements.txt
```

## Relased Pre-trained Model (Folder "model/released")
The pre-trained CircuitFusion model has been released. We will maintain this general CircuitFusion model and further pre-train it on more unlabeled circuit samples.

 
## Circuit Data Collection (Folder: "dara_collect")

* All the RTL designs used in our work are collected from open-source projects, their links are attached below:
    ```
    ITC99: https://iwls.org/iwls2005/benchmarks.html
    OpenCores: https://opencores.org/
    VexRiscv: https://github.com/SpinalHDL/VexRiscv
    Chipyard: https://github.com/ucb-bar/chipyard
    ```
* The RTL designs are synthesized using Synopsys Design Compiler into netlists
* Data augmentation
  - The functionally equivalent circuit designs are generated through equivalent transformations using synthesis tools (e.g., Yosys+ABC, Design Compiler). Examples are shown in the "ori" and "pos" folders in "dataset/rtl" and "dataset/netlist"
  - The non-equivalent circuits are augmented by randomly selecting other designs, with examples shown in the "neg" folders




## Multimodal and Multi-Stage Circuit Preprocessing and Alignment (Folder: "dara_collect")

* Circuit-to-graph transformation
  - The RTL code and netlist are converted into graph format based on Verilog parsers
  - RTL transformation: vlg_design2graph.py in folder "rtl/scr_design2graph"
  - Netlist transformation: net_design2graph.py in folder "netlist/scr"
* Cross-stage circuit alignment
  - The circuits are aligned through registers.
  - Each register backtrace all the logic until reaching all other registers --> cone of register
  - RTL cone extraction: vlg_cone2graph.py in folder "rtl/scr_cone2graph"
  - Netlist cone extraction: net2subgraph.py in folder "netlist/scr"
    


## CircuitFusion Pre-training (Folder: "model/ckt_fusion")

### Pre-training 
```
cd ./model/ckt_fusion
python3 pretrain_align_all.py
```
 
### CircuitFusion model
- Folder "/model/ckt_fusion/models"
- Change configurations in "/model/ckt_fusion/configs"
- Model saved in "/model/ckt_fusion/pretrain_model"

## Evaluation on Downstream Tasks (Folder: "model/ckt_fusion")

### Cross-Stage Circuit Retrieval & Zero-Shot Inference 
* Retrieval based on the similarities between circuit embeddings
```
cd ./model/ckt_fusion
python3 infer.zero_shot.py
```

### Few-Shot Inference 
* Fine-tuning for downstream tasks
* Supervised training based on task-specific labels:
  ```
  cd ./model/ckt_fusion
  python3 finetune_slack.py
  python3 finetune_wns.py
  python3 finetune_tns.py
  python3 finetune_power.py
  python3 finetune_area.py
  ```


## Abstract

The rapid advancements of AI rely on the support of integrated circuits (ICs). However, the growing complexity of digital ICs makes traditional IC design process costly and time-consuming. In recent years, AI-assisted IC design methods have demonstrated great potential. However, most existing methods are task-specific or focus solely on the circuit structure in graph format, overlooking other circuit modalities that contain rich functional information. In this paper, we introduce CircuitFusion, the first multimodal and implementation-aware circuit encoder. It encodes circuits into general representations that support different downstream design tasks. To learn from circuit, we propose to fuse three circuit modalities: hardware code, structural graph, and functionality summary. Moreover, we identify four unique properties of circuits: parallel execution, functional equivalent transformation, multiple design stages, and circuit reusability. Based on these properties, we propose new strategies during the development and application of CircuitFusion: 1) During circuit preprocessing, utilizing the parallel nature of circuits, we split each circuit into multiple sub-circuits based on sequential element boundaries. It enables fine-grained encoding at the sub-circuit level. 2) During CircuitFusion pre-training, we introduce three self-supervised tasks that utilize equivalent transformations both within and across modalities. We further utilize the multi-stage design process to align representation with ultimate circuit implementation. 3) When applying CircuitFusion to downstream tasks, we propose a new retrieval-augmented inference method, which retrieves similar known circuits as a reference for predictions. It improves fine-tuning performance and even enables zero-shot inference. Evaluated on five different IC design tasks, CircuitFusion consistently outperforms the state-of-the-art supervised method specifically developed for each task, demonstrating its generalizability and ability to learn circuits' fundamental inherent properties.