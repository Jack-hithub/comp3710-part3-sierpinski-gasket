# Local CPU benchmark evidence

## Environment

- Date: 2026-08-21
- Computer: MacBook Air, Apple M3, 8 CPU cores, 16 GB memory
- Operating system: macOS 15.7.3
- Python: 3.11.15
- PyTorch: 2.13.0
- Device: CPU

Hardware serial numbers and other unique machine identifiers are deliberately
excluded because they are irrelevant to reproducibility.

## Method

The command used was:

```bash
python src/benchmark_gasket.py \
  --sizes 256 512 1024 2048 4096 \
  --devices cpu \
  --warmups 3 \
  --repeats 9
```

Each timing includes PyTorch generation of the Boolean `rows x rows` mask and
the centred `rows x (2*rows-1)` raster. It excludes plotting and the final
transfer to NumPy. The three warm-up runs per size are discarded, followed by
nine measured runs. Median time is the main summary because it is less
sensitive to occasional system activity than the mean.

## Results

| Device | Rows | Median ms | Mean ms | Std. dev. ms | Candidate MPixels/s | Points |
|---|---:|---:|---:|---:|---:|---:|
| CPU | 256 | 0.219 | 0.226 | 0.014 | 299.59 | 6,561 |
| CPU | 512 | 0.387 | 0.409 | 0.057 | 678.03 | 19,683 |
| CPU | 1,024 | 1.268 | 1.316 | 0.134 | 826.79 | 59,049 |
| CPU | 2,048 | 4.257 | 4.450 | 0.424 | 985.28 | 177,147 |
| CPU | 4,096 | 15.608 | 15.903 | 0.998 | 1,074.89 | 531,441 |

The exact per-run values are stored in `outputs/benchmark_results.csv`.

## Interpretation

Median generation time increases from 0.219 ms at 256 rows to 15.608 ms at
4,096 rows. Candidate-grid throughput also rises with problem size because
fixed Python and allocation overhead becomes less important. This experiment
is not presented as a hardware comparison; it demonstrates repeatable scaling
of the vectorised PyTorch implementation on the CPU available for the demo.

The foreground point count follows the exact `3^n` rule for every tested
power-of-two row count, so the timing experiment also provides a correctness
check at increasing resolutions.
