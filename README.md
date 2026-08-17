# COMP3710 Lab 1 Part 3 - Sierpinski Gasket

This repository develops the **Sierpinski gasket by binary addresses** from
Section 2.10 of *Fractals for the Classroom*. It is a new Part 3 project and
does not reuse the earlier Sierpinski carpet proposal.

## Current status

- Fractal choice: confirmed from a textbook section explicitly suggested by
  the lab sheet.
- Development stage: project preparation and independent reference method.
- Remote repository: none. GitHub will only be connected after the student
  creates or approves the destination repository.
- Final PyTorch implementation: deliberately not started yet. The project is
  being built and reviewed in small commits so the learning process remains
  visible.

## Planned evidence

1. A small NumPy/Python reference based on the parity of Pascal's triangle.
2. Independent checks using exact binomial coefficients and known point
   counts.
3. A reviewed PyTorch implementation in which pixels are evaluated in
   parallel on CPU, Apple MPS, or NVIDIA CUDA.
4. CPU/GPU timing at several image sizes.
5. A box-counting estimate compared with the theoretical dimension
   `log(3) / log(2)`.
6. More than one visualisation, with parameters and colours explained.
7. Course-required AI-use documentation maintained separately by the student.

## Local environment

The existing `comp3710` conda environment can be reused:

```bash
conda activate comp3710
cd /Users/baoxuwang/Documents/3710/part3_sierpinski_gasket
```

To recreate it later:

```bash
conda env create -f environment.yml
```

Run the completed reference stage with:

```bash
python tests/verify_reference.py
python src/reference_gasket.py --rows 128
```

The first command compares the binary-address result with exact binomial
coefficients. The second writes `outputs/reference_gasket.png`.

## AI documentation ownership

The student will maintain and organise the course-required AI-use record. The
project automation will not create or update that record unless the student
explicitly requests it later.

## Reference-stage result

For 128 (`2^7`) rows, the reference contains 2,187 (`3^7`) odd Pascal
entries. Its centred raster has shape `128 x 255`. The generated figure has
been visually reviewed and shows the expected recursive triangular gaps.
