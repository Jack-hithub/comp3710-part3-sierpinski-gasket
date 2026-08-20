# AI Prompts

## Prompt 1 - Understanding the requirements

I understand that Part 3 requires a fractal implemented with PyTorch,
TensorFlow, or JAX. I am considering the Sierpinski gasket from Section 2.10
of the suggested textbook.

Before discussing implementation, help me identify what must be demonstrated,
what evidence of parallel tensor computation is needed, what analysis would be
considered substantial, and which decisions I still need to make myself. Do not
generate the project or complete code yet.

## Prompt 2 - Understanding the mathematics

The textbook connects the Sierpinski gasket with the parity of Pascal's
triangle and gives the binary condition:

```text
c & (r - c) == 0
```

Explain why this condition identifies odd binomial coefficients. Work through
several small examples by hand, including one odd and one even coefficient.
Focus on the mathematics and do not write the implementation yet.

## Prompt 3 - Reviewing the algorithm

My current algorithm is:

1. create row and column coordinates;
2. keep only positions where `column <= row`;
3. apply the binary parity condition; and
4. map the triangular mask to a centred image.

Review this plan for mathematical or indexing mistakes. Ask me questions if an
important design decision is missing. Do not replace it with a complete
program.

## Prompt 4 - Checking the reference implementation

I have written a small NumPy reference implementation based on the algorithm
above. I plan to compare its results with exact values from `math.comb`.

Review the function I provide and suggest a small set of correctness tests. In
particular, consider symmetry, small Pascal rows, foreground point counts, and
whether the display transformation loses any points. Explain why each test is
useful instead of rewriting the whole function.

## Prompt 5 - Converting the core operation to PyTorch

My NumPy reference is working. I now want to express only the core pixel test
using PyTorch broadcasting.

My proposed tensor shapes are:

```text
row:    (N, 1)
column: (1, N)
```

Explain what output shape these produce, how the triangular boundary and
bitwise condition can be applied, and why this counts as parallel tensor
computation. Show only the essential tensor expression, not a complete
repository or plotting program.

## Prompt 6 - Verifying the PyTorch result

My PyTorch output visually resembles a Sierpinski gasket, but I do not want to
treat appearance as proof of correctness.

Help me design tests that compare it with my independent NumPy reference and
the known count `3^n` for the first `2^n` rows. Also identify possible dtype,
device-placement, and display-mapping errors. Give me a testing strategy rather
than generating replacement code.

## Prompt 7 - Selecting substantial analysis

I need analysis beyond producing a fractal image. I am considering box-counting
dimension, comparison with `log(3) / log(2)`, and colouring points using
information from their binary addresses.

Explain what each analysis demonstrates and what limitations I should mention.
Help me decide which results should appear in the final figure, but leave the
final design choice to me.

## Prompt 8 - Reviewing the benchmark

I propose benchmarking several image sizes on CPU using three warm-up runs and
nine measured repetitions. I will save every measurement, report the median,
mean, and standard deviation, and calculate throughput using the `N x N`
candidate grid.

Critique this method. Explain what the benchmark can support and what claims it
cannot support. Do not invent performance results.

## Prompt 9 - Choosing the execution environment

I originally prepared a Google Colab and CUDA workflow because I thought remote
GPU execution might be expected. The teacher has now confirmed that CPU
execution is acceptable.

Compare the advantages and disadvantages of keeping the Colab material versus
submitting a local CPU demonstration. My priority is that the workflow is
simple, reproducible, and accurately reflects how the final figures were
generated.

## Prompt 10 - Auditing the repository

My code and tests are committed, but I am concerned that generated figures and
CSV results may have been excluded by `.gitignore`.

Give me a checklist for auditing the GitHub repository before the demonstration.
Include source code, tests, documentation, result images, raw measurements,
reproducibility instructions, and AI-use documentation. Do not modify or push
anything without my confirmation.
