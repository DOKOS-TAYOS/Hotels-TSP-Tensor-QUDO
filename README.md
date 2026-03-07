# Aircraft Loading Problem (T-QUDO)

Aircraft cargo loading optimization using Tensor-QUDO (Tensor Quadratic Unconstrained Discrete Optimization) and quantum computing backends.

## Structure

```
src/
├── cirq/           # Cirq quantum backend
├── cudaq/          # NVIDIA CUDA Quantum backend
├── config/         # Configuration management
├── instance_generator/  # Problem instance generation
└── streamlit_app/  # Web UI
```

## Setup

```bash
pip install -e .
```

## Run

```bash
streamlit run src/streamlit_app/app.py
```

## Configuration

- Copy `.env.example` to `.env` and adjust
- Edit `src/instance_generator/config.yaml` for instance parameters
