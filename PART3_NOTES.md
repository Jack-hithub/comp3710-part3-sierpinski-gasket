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
construction all run on the selected PyTorch device. The submitted demo uses
CPU, while the code can optionally select MPS or CUDA when available. Only the
final raster moves to CPU:

```python
image_array = raster.cpu().numpy()
```

That final conversion is necessary because Matplotlib expects a NumPy array.
Moving intermediate tensors between devices would add unnecessary transfer
overhead.

## 5. How correctness is checked

The PyTorch result is not accepted just because its picture looks plausible.
`tests/verify_torch_gasket.py` checks:

- the known first four Pascal rows and their centred display positions;
- known counts `3^n` for the first `2^n` rows;
- Boolean mask and unsigned-byte raster dtypes;
- preservation of the selected device;
- invalid input handling; and
- equality after mapping to the centred raster.

The submitted CPU implementation passes all of these checks locally.

## 6. Box-counting dimension

Box counting covers the foreground pixels with square boxes. If a box has
side length `s`, the program counts how many such boxes contain at least one
gasket point. It then fits:

```text
log(number of occupied boxes) = D * log(image width / s) + constant
```

The fitted slope `D` estimates the fractal dimension.

For an image with `N = 2^n` rows and aligned boxes of size `2^k`, the gasket
contains exactly `3^(n-k)` occupied boxes. Every factor-of-two decrease in box
width reveals three self-similar copies. Therefore:

```text
D = log(3) / log(2) = 1.5849625007...
```

The 1,024-row experiment measured the following sequence:

```text
box size:        1      2     4    8   16  32  64  128  256
occupied:    59049  19683  6561 2187  729 243  81   27    9
```

The fitted value was `1.584962501` with `R^2 = 1.0`. This exact alignment is
expected for this deterministic gasket and these power-of-two scales; it is
not evidence that all empirical fractal-dimension estimates will be exact.

The analysis uses the right-triangular Pascal mask so boxes align naturally
with the binary construction. The centred display is a shear and scale of the
same set and does not change its theoretical dimension.

## 7. Meaning of the address colours

Every foreground point has two non-overlapping binary addresses, `c` and
`row-c`. The colour view counts the 1-bits in each address and plots their
difference. Blue points use more bits from `row-c`, red points use more bits
from `c`, and pale points are balanced. This adds information about the binary
construction instead of applying arbitrary decorative colours.
