[README.md](https://github.com/user-attachments/files/24397135/README.md)
# Predictive Maintenance Under Prognostic Uncertainty

**The Marginal Value of RUL Accuracy in Fleet Operations**

This repository contains the simulation codebase for the paper *"Predictive Maintenance Under Prognostic Uncertainty: The Marginal Value of RUL Accuracy in Fleet Operations."*

## Overview

This paper quantifies the marginal value of remaining useful life (RUL) prediction accuracy in capacity-constrained fleet operations under stochastic demand. Using a stochastic simulator calibrated to U.S. Army rotorcraft maintenance doctrine, we optimize decision-tree policies via a heterogeneous island-model genetic algorithm across five accuracy levels, measured by the coefficient of variation (CV) of the RUL signal.

### Key Findings

- **Diminishing returns**: 79% of achievable improvement in mission success is captured by CV=25%, and 98% by CV=10%
- **Sensor accuracy dominates**: The factorial analysis reveals that sensor accuracy—not policy sophistication—drives performance gains
- **Information quality matters**: Policies trained under noisy conditions perform comparably to those trained under accurate conditions when both receive the same signal

### Main Contribution

CV is the sufficient statistic for decision relevance in capacity-constrained fleet operations: if an organization knows its prognostic CV, it knows whether further sensor investment, modeling effort, or policy redesign will yield operational benefit.

## Repository Structure

```
├── simulation/       # Core simulation engine
│   ├── environment.py        # Main simulation loop
│   ├── aircraft.py           # Aircraft state management
│   ├── mission_generator.py  # Stochastic mission demand
│   ├── maintenance_system.py # Maintenance queue processing
│   └── ...
├── optimization/     # Genetic algorithm implementation
│   ├── island_ga.py          # Island model GA (Oahu→Maui→Big Island)
│   ├── ga_algorithm.py       # Standard GA implementation
│   ├── fitness_evaluator.py  # Multi-objective fitness
│   └── ...
├── policy/          # Policy representations
│   ├── chromosome.py         # Genome encoding/decoding
│   ├── decision_tree_policy.py # Decision tree policy
│   └── ...
├── config/          # Configuration files
│   ├── ga/                   # GA hyperparameters
│   ├── features/             # Feature set definitions
│   └── scenarios/            # Simulation parameters
├── scripts/         # Entry point scripts
│   ├── run_island_ga.py      # Run island model optimization
│   ├── run_ga.py             # Run standard GA
│   └── ...
├── tests/           # Test suite
└── utils/           # Utility functions
```

## Installation

```bash
# Clone the repository
git clone https://github.com/defense031/preventive_maintenance_us_army_aviation_genetic_algorithm.git
cd preventive_maintenance_us_army_aviation_genetic_algorithm

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Island Model GA Optimization

```bash
python scripts/run_island_ga.py --config config/ga/default.yaml
```

### Run Standard GA

```bash
python scripts/run_ga.py --config config/ga/default.yaml
```

### Run Baseline Evaluation

```bash
python scripts/run_baseline.py
```

## Configuration

### Simulation Parameters (`config/scenarios/default.yaml`)
- Fleet size, maintenance capacity
- Mission demand distributions
- RUL initialization and decay
- Maintenance duration distributions

### GA Parameters (`config/ga/default.yaml`)
- Population size, generations
- Crossover and mutation rates
- Island migration topology
- Fitness weights (mission success vs. operational readiness)

### Feature Sets (`config/features/`)
- `simple.yaml`: Minimal features for fast iteration
- `medium.yaml`: Balanced feature set
- `full.yaml`: Complete feature engineering

## Citation

If you use this code in your research, please cite:

```bibtex
@phdthesis{semmel2025,
  title={Predictive Maintenance Under Prognostic Uncertainty: The Marginal Value of RUL Accuracy in Fleet Operations},
  author={Semmel, Austin},
  year={2025},
  school={North Carolina State University}
}
```

## License

This project is released for academic and research purposes. See LICENSE file for details.

## Contact

For questions about the code or methodology, please open an issue on this repository.
