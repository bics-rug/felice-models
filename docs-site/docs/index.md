# Felice

This project provides a [JAX](https://github.com/google/jax) implementation of the different neuron models in felice

## Overview

The framework is built on top of diffrax and leverages JAX's automatic differentiation for efficient simulation and training of analogue models.

### Key Features

- **Delay learning**
- **Non-linear neuron models**
    - [**WereRabbit Neuron Model**](neuron_models/wererabbit/index.md): Implementation of a dual-state oscillatory neuron model with bistable dynamics
    - [**FHN Neuron Model**](neuron_models/fhn/index.md)
    - [**Snowball Neuron Model**](neuron_models/snowball/index.md)

## 📦 Installation

Felice uses [uv](https://github.com/astral-sh/uv) for dependency management. To install:

```bash
uv sync
```

### CUDA Support (Optional)

For GPU acceleration with CUDA 13:

```bash
uv sync --extra cuda
```

See the [examples](https://github.com/bics-rug/felice-models/tree/main/scripts/examples/neuron_models) directory for more detailed usage examples.