# Felice

This project provides a [JAX](https://github.com/google/jax) implementation of the different neuron models in felice

## Overview

The framework is built on top of diffrax and leverages JAX's automatic differentiation for efficient simulation and training of analogue models.

### Key Features

- **Delay learning**
- **WereRabbit Neuron Model**: Implementation of a dual-state oscillatory neuron model with bistable dynamics

## Installation

Felice uses [uv](https://github.com/astral-sh/uv) for dependency management. To install:

```bash
uv sync
```

### CUDA Support (Optional)

For GPU acceleration with CUDA 13:

```bash
uv sync --extra cuda
```

## Neuron Models

### WereRabbit

The WereRabbit neuron model implements a bistable oscillatory system with dual-state dynamics (x1, x2). Key characteristics:

- **Bistable dynamics**: Two stable states with smooth transitions
- **Event-based spiking**: Spikes detected when system reaches a fixpoint
- **Configurable parameters**: Bias current, scaling distance, and tolerance settings for the fixpoint calculation

Configuration parameters:
- `Ibias`: Bias current (default: 300e-12 A)
- `scaling_distance`: Scaling factor for state attraction (default: 0.6)
- `rtol`, `atol`: Relative and absolute tolerances for spike detection

## Development

### Running Tests

```bash
uv run pytest tests
```

### Building Documentation

Documentation is built with MkDocs:

```bash
cd docs-site && uv run mkdocs serve
```

### Code Quality

The project uses pre-commit hooks for code formatting and linting with Ruff:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

## Requirements

- Python >= 3.13
- JAX >= 0.8.1
- Equinox >= 0.13.2
- Diffrax >= 0.7.0

See [pyproject.toml](pyproject.toml) for the complete list of dependencies.

## Contributing

Contributions are welcome! Please ensure:
1. Code passes all tests (`pytest`)
2. Code is formatted with Ruff (`pre-commit run --all-files`)
3. Type annotations are included for new functions

## License

- **Code** is licensed under the [MIT License](./LICENSE).
- **Documentation** is licensed under [CC BY 4.0](./docs-site/LICENSE).