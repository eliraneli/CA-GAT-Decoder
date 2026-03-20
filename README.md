# CA-GAT-NMS: Cycle-Aware Graph Attention Neural Min-Sum Decoder

> **Official PyTorch Implementation**

This repository contains the source code for the **Cycle-Aware Graph Attention Neural Min-Sum (CA-GAT-NMS)** decoder. This architecture addresses the severe performance degradation caused by short cycles in the Tanner graphs of short block-length Forward Error Correction (FEC) codes, specifically focusing on short 5G NR LDPC codes and dense BCH codes.

By combining an offline graph-traversal algorithmic mask with an online Graph Attention Network (GAT), CA-GAT-NMS dynamically routes messages along cycle-free paths, significantly lowering the error floor while maintaining the hardware-friendly O(V + E) time complexity of traditional Belief Propagation (BP).

## Features
* **Custom Environment:** Integrates with NVIDIA Sionna for AWGN channel simulation.
* **Database Support:** Automatically parses standard `.alist` and `.npy` parity-check matrices.
* **Algorithmic Pre-processing:** BFS-based offline short-cycle detection and masking.
* **Multi-Head GAT:** Configurable Lite (1-head) and True (4-head, 16-dim) Graph Attention Networks.
* **Hardware-Friendly Options:** Toggles for shared weights (RNN-style) and early stopping via syndrome checking.
* **Publication Plotting:** Automated generation of vector (.pdf) and raster (.png) graphs for BER and FER.

## Installation
Ensure you have Python 3.8+ installed. Install the required dependencies using:
`pip install -r requirements.txt`

## Quick Start
To train the model and generate performance graphs, simply run:
`python main.py`
