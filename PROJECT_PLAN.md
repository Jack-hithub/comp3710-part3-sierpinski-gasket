# Part 3 staged plan

The stages below are intentionally separated so that the Git history,
verification, and experiment notes show an understandable development process.

## Stage 0 - selection and preparation

- [x] Replace the unapproved Sierpinski carpet proposal.
- [x] Select the Sierpinski gasket from textbook Section 2.10.
- [x] Confirm the existing `comp3710` environment.
- [x] Create a clean local repository and connect the student-approved remote.

## Stage 1 - independent reference

- [x] Implement Pascal-triangle parity with NumPy/Python.
- [x] Check it against exact binomial coefficients.
- [x] Verify symmetry and the count `3^n` in the first `2^n` rows.
- [x] Inspect a small reference image.

## Stage 2 - reviewed PyTorch implementation

- [x] Express the binary-address test with PyTorch integer tensors.
- [x] Add explicit CPU, MPS, and CUDA device selection.
- [x] Compare the PyTorch mask with the independent reference.
- [x] Explain tensor shapes, bitwise operations, and device movement.

## Stage 3 - substantial analysis

- [x] Benchmark several resolutions on CPU and Apple MPS.
- [x] Estimate the box-counting dimension.
- [x] Compare the estimate with `log(3) / log(2)`.
- [x] Add labelled binary and address-colour visualisations.

## Stage 4 - CUDA and submission evidence

- [ ] Run the reviewed code in Google Colab with CUDA.
- [ ] Save environment information, logs, timings, and figures.
- [ ] Review every retained output.
- [x] Connect the GitHub remote and push initial commits after student approval.
