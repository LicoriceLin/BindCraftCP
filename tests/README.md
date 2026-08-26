# Tests

Run tests from an interactive Slurm compute allocation:

```bash
conda activate BindCraft
python -m pytest
```

Contract tests use lightweight instrumented Steps. Tests marked `gpu` invoke a
real backend and run by default when an NVIDIA GPU and the required conda
environment are available; they skip only when that environment is absent.
The current BoltzGen smoke test uses one recycle, ten sampling steps, and one
diffusion sample to validate the complete Runner-to-CIF mapping path.
