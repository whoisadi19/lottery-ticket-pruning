# Lottery Ticket Hypothesis - Neural Network Pruning

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Aditya Chowdhury  
**Institution:** NMIMS MPSTME Shirpur  
**Course:** Computer Science  
**Date:** January 2026

##  Project Overview

This project implements and validates MIT's **Lottery Ticket Hypothesis**, demonstrating that neural networks can be pruned by **90% while maintaining (and even improving!) accuracy**. 

### Key Finding
- **90% pruning** → 97.94% accuracy (improvement over baseline!)
- **80% pruning** → 98.29% accuracy (best performance!)
- **10× smaller model** with better performance

##  Research Article

This implementation is part of a research article submitted to **NMIMS Tech Trends** technical magazine. The complete article explores:
- The Lottery Ticket Hypothesis theory
- Practical implementation details
- Experimental results and analysis
- Real-world implications for AI efficiency

##  Quick Start

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/whoisadi19/lottery-ticket-pruning.git
cd lottery-ticket-pruning

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Experiment

**Option 1: Direct execution**
```bash
# Run the pruning experiment
python lottery_ticket_pruning.py

# Generate visualizations
python visualize_results.py
```

**Option 2: Using helper script (suppresses warnings)**
```bash
# Cleaner output without numpy warnings
python run_experiment.py
```

**Option 3: Test your setup first**
```bash
# Verify PyTorch installation
python test_setup.py
```

**Expected Runtime:** 15-20 minutes on CPU

##  Results

### Experimental Setup
- **Dataset:** MNIST (70,000 handwritten digits)
- **Architecture:** 784 → 300 → 100 → 10 (fully connected)
- **Training:** 5 epochs per model
- **Pruning Levels:** 0%, 20%, 40%, 60%, 70%, 80%, 90%, 95%

### Key Results

| Pruning % | Parameters | Accuracy | Change |
|-----------|-----------|----------|--------|
| Baseline  | 266,610   | 97.48%   | -      |
| 90%       | 26,617    | 97.94%   | +0.46% |
| 80%       | 53,237    | 98.29%   | +0.81% |

**Surprising Discovery:** Pruning actually *improves* accuracy by removing redundant parameters that add noise!

##  Project Structure

```
lottery-ticket-pruning/
├── lottery_ticket_pruning.py   # Main implementation
├── visualize_results.py        # Visualization generation
├── run_experiment.py           # Helper script with warnings suppressed
├── test_setup.py               # Quick setup verification
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── results/
│   └── experiment_results.json # Experimental data
└── images/
    ├── accuracy_vs_pruning.png
    ├── parameter_reduction.png
    ├── accuracy_difference.png
    ├── combined_metrics.png
    └── results_table.png
```

##  Implementation Details

### The Lottery Ticket Algorithm

1. **Initialize:** Create a neural network with random weights (save initial values!)
2. **Train:** Train the full network to convergence
3. **Prune:** Remove weights with smallest magnitudes
4. **Reset:** Reset remaining weights to their original initialization
5. **Retrain:** Train the sparse network from the original initialization

### Key Features

- ✅ Magnitude-based pruning
- ✅ Proper initialization handling
- ✅ Mask maintenance during training
- ✅ Comprehensive logging
- ✅ Result visualization

##  Visualizations

The project generates 5 professional visualizations:

1. **Accuracy vs Pruning** - Shows accuracy remains high across pruning levels
2. **Parameter Reduction** - Dramatic visualization of model compression
3. **Accuracy Difference** - Highlights performance improvements
4. **Combined Metrics** - Dual-axis view of accuracy and model size
5. **Results Table** - Professional summary of all results

##  Educational Value

This project demonstrates:
- Deep learning fundamentals
- Neural network pruning techniques
- PyTorch implementation skills
- Experimental methodology
- Data visualization
- Scientific writing

##  References

1. Frankle, J., & Carbin, M. (2019). "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks." *ICLR 2019*.

2. Han, S., Pool, J., Tran, J., & Dally, W. (2015). "Learning both Weights and Connections for Efficient Neural Network." *NeurIPS 2015*.



##  Contact

**Aditya Chowdhury**
- Email: faithfullyours.adi@gmail.com
- GitHub: [@whoisadi19](https://github.com/whoisadi19)
- Institution: NMIMS MPSTME Shirpur


##  Acknowledgments

- MIT researchers Jonathan Frankle and Michael Carbin for the original Lottery Ticket Hypothesis


##  Star This Repository

If you find this project helpful or interesting, please consider giving it a star! It helps others discover this work.

---

**Made with ❤️ for advancing AI efficiency research**
