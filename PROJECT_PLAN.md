# Part 3 staged plan

The stages below are intentionally separated so that the Git history and AI
log show an understandable development process.

## Stage 0 - selection and preparation

- [x] Replace the unapproved Sierpinski carpet proposal.
- [x] Select the Sierpinski gasket from textbook Section 2.10.
- [x] Confirm the existing `comp3710` environment.
- [x] Create a clean local repository with no GitHub remote.

## Stage 1 - independent reference

- [x] Implement Pascal-triangle parity with NumPy/Python.
- [x] Check it against exact binomial coefficients.
- [x] Verify symmetry and the count `3^n` in the first `2^n` rows.
- [x] Inspect a small reference image.

## Stage 2 - reviewed PyTorch implementation

- [ ] Express the binary-address test with PyTorch integer tensors.
- [ ] Add explicit CPU, MPS, and CUDA device selection.
- [ ] Compare the PyTorch mask with the independent reference.
- [ ] Explain tensor shapes, bitwise operations, and device movement.

## Stage 3 - substantial analysis

- [ ] Benchmark several resolutions on available devices.
- [ ] Estimate the box-counting dimension.
- [ ] Compare the estimate with `log(3) / log(2)`.
- [ ] Add labelled visualisations that reveal construction depth or scale.

## Stage 4 - CUDA and submission evidence

- [ ] Run the reviewed code in Google Colab with CUDA.
- [ ] Save environment information, logs, timings, and figures.
- [ ] Review every retained output.
- [ ] Connect and push to GitHub only after student approval.
