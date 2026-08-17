# Part 3 understanding notes

## 1. What is being drawn?

The Sierpinski gasket appears when Pascal's triangle is coloured by parity:
odd binomial coefficients are foreground points and even coefficients are
background. This is different from the square Sierpinski carpet.

For Pascal row `r` and column `c`, the binomial coefficient is odd exactly
when:

```text
c & (r - c) == 0
```

Here `&` is bitwise AND, not the Boolean word `and`.

Two small examples:

- Row 5, column 1: `001 & 100 = 000`, so `C(5, 1) = 5` is odd.
- Row 4, column 2: `010 & 010 = 010`, so `C(4, 2) = 6` is even.

## 2. Why is the PyTorch calculation parallel?

For `N` rows, the implementation creates one address tensor of shape `(N,)`
and views it in two ways:

```text
row:    (N, 1)
column: (1, N)
```

PyTorch broadcasting evaluates them as an `(N, N)` grid. There is no Python
loop over individual pixels. The same comparison and bitwise expression are
applied to all possible `(row, column)` positions by tensor operations.

The triangular boundary check `column <= row` removes positions that do not
belong to Pascal's triangle. The result has Boolean dtype because each entry
answers a yes/no question.

## 3. Why is there a separate centred raster?

The mathematical mask stores Pascal row `r` in the first `r + 1` columns.
That is convenient for checking binomial coefficients but appears as a
right-angled triangle. For display, a foreground position `(r, c)` is mapped
to:

```text
display_x = (N - 1 - r) + 2*c
```

This centres every row and produces the familiar equilateral-triangle layout.
The verification checks that this display mapping neither loses nor creates
points.

## 4. Device movement

`torch.arange`, the broadcast comparison, the bitwise expression, and raster
construction all run on the selected CPU, MPS, or CUDA device. Only the final
raster moves to CPU:

```python
image_array = raster.cpu().numpy()
```

That final conversion is necessary because Matplotlib expects a NumPy array.
Moving intermediate tensors back and forth would add transfer overhead and
would weaken the GPU experiment.

## 5. How correctness is checked

The PyTorch result is not accepted just because its picture looks plausible.
`tests/verify_torch_gasket.py` checks:

- exact equality with the independent NumPy implementation;
- known counts `3^n` for the first `2^n` rows;
- Boolean mask and unsigned-byte raster dtypes;
- preservation of the selected device;
- invalid input handling; and
- equality after mapping to the centred raster.

Both CPU and Apple MPS passed these checks locally. NVIDIA CUDA will be checked
later in Colab.

## 6. Timing caution

Accelerators initialise lazily, so the first MPS or CUDA operation can include
startup and compilation overhead. A single printed time is useful diagnostic
information but not a fair benchmark. The analysis stage will use warm-up
runs, synchronisation, repeated trials, and summary statistics.

