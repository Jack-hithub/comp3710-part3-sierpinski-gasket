# Local benchmark evidence

## Environment

- Date: 2026-08-18
- Computer: MacBook Air, Apple M3, 8 CPU cores, 16 GB memory
- Operating system: macOS 15.7.3
- Python: 3.11.15
- PyTorch: 2.13.0
- Compared devices: CPU and Apple MPS

Hardware serial numbers and other unique machine identifiers are deliberately
excluded because they are irrelevant to reproducibility.

## Method

The command used was:

```bash
python src/benchmark_gasket.py \
  --sizes 256 512 1024 2048 4096 \
  --devices available \
  --warmups 3 \
  --repeats 9
```

Each timing includes PyTorch generation of the Boolean `rows x rows` mask and
the centred `rows x (2*rows-1)` raster. It excludes plotting and the final
transfer to NumPy. Accelerator operations are synchronised before and after
the timed region. The three warm-up runs per size and device are discarded.

## Results

| Device | Rows | Median ms | Mean ms | Std. dev. ms | Candidate MPixels/s | Points |
|---|---:|---:|---:|---:|---:|---:|
| CPU | 256 | 0.214 | 0.222 | 0.016 | 306.12 | 6,561 |
| CPU | 512 | 0.349 | 0.375 | 0.055 | 750.14 | 19,683 |
| CPU | 1,024 | 1.212 | 1.331 | 0.233 | 865.13 | 59,049 |
| CPU | 2,048 | 4.161 | 4.135 | 0.222 | 1,007.93 | 177,147 |
| CPU | 4,096 | 15.439 | 15.927 | 1.221 | 1,086.69 | 531,441 |
| MPS | 256 | 1.033 | 0.968 | 0.152 | 63.43 | 6,561 |
| MPS | 512 | 0.947 | 0.994 | 0.215 | 276.82 | 19,683 |
| MPS | 1,024 | 4.436 | 3.415 | 1.533 | 236.37 | 59,049 |
| MPS | 2,048 | 4.724 | 5.013 | 0.520 | 887.87 | 177,147 |
| MPS | 4,096 | 17.746 | 17.726 | 0.056 | 945.41 | 531,441 |

The exact per-run values are written to `outputs/benchmark_results.csv` when
the benchmark is run.

## Interpretation

CPU has the lower median at every tested size. At 256 rows MPS is dominated by
fixed launch overhead. By 2,048 and 4,096 rows the two devices are much closer,
but CPU remains faster on this machine. The project therefore reports the
measured result rather than assuming that GPU execution must be faster.

CUDA is a different backend and hardware platform. The same controlled method
will be repeated in Google Colab before drawing any conclusion about CUDA.
