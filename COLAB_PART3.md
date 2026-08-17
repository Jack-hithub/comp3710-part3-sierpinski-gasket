# Part 3 CUDA run in Google Colab

The Colab notebook runs the existing reviewed project. It does not generate or
replace the implementation.

## Before opening Colab

From the project directory, build the upload bundle:

```bash
./scripts/build_colab_bundle.sh
```

This creates:

```text
dist/COMP3710_Part3_Colab_Bundle.zip
```

## Colab steps

1. Upload `COMP3710_Part3_Colab.ipynb` to Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU** and save.
3. Run the notebook cells one at a time.
4. In the upload cell, select `dist/COMP3710_Part3_Colab_Bundle.zip`.
5. Read each test and experiment result before moving on.
6. In the final cell, download `COMP3710_Part3_Colab_Evidence.zip`.

Keep the downloaded evidence ZIP. It contains the CUDA environment report,
stdout/stderr, verified source files, benchmark CSV, and CUDA figures.

## Expected checkpoints

- `CUDA available: True`
- a visible NVIDIA GPU name, normally Tesla T4 on the free tier
- `Verified device: cuda` in the PyTorch checks
- a successful CUDA gasket run
- an estimated dimension close to `1.584962501`
- CPU and CUDA benchmark rows for all five image sizes

Exact timings and assigned GPU model can vary between Colab sessions.

