# AI Use Declaration and Development Record

## Declaration

I used OpenAI ChatGPT/Codex as an AI-assisted programming and discussion tool
for this project. Its roles included helping me interpret the task, organise a
staged development plan, explain the mathematics, draft and review code, design
tests, interpret experimental results, and improve the documentation.

The project was not produced from one prompt. I developed it through a sequence
of questions, checks, corrections, and decisions. I reviewed the suggestions,
ran the code on my own Mac, inspected the generated figures, and retained or
changed the output according to the course requirements and the evidence from
the tests. The final repository therefore contains AI-assisted work that I have
personally examined and can explain.

## Development process and AI interaction record

The prompts below are concise records of the requests made during development.
Some were originally asked in Chinese; they are summarised in English here so
that the technical progression is easy to follow. The response column records
the useful content of the AI output, while the final column records my own
decision, modification, or verification. This is a structured record rather
than a claim that the AI's first response was accepted unchanged.

| Stage and prompt/request | Main AI output or reasoning | My review, decision, and modification |
| --- | --- | --- |
| **1. Interpret the assessment:** What does the lab require, and does it require remote GPU training? | The AI separated the instructions in the lab sheet from optional computing arrangements. It explained that Part 3 requires a PyTorch implementation and meaningful analysis, but that the fractal calculation is not neural-network training and can be demonstrated on CPU if permitted by the teaching staff. | I checked this interpretation against the lab PDF and the teacher's clarification. I chose a local CPU workflow so that the demo is reproducible on my Mac. |
| **2. Choose a suitable fractal:** Find a Part 3 topic that is different from the Mandelbrot and Julia sets and is supported by the suggested textbook. | The AI proposed the Sierpinski gasket by binary addresses from Section 2.10 of *Fractals for the Classroom*. It explained the connection with odd entries in Pascal's triangle and the condition `c & (r - c) == 0`. | I accepted this topic because it comes from a section explicitly suggested in the task. I recorded the textbook and lab-sheet sources in `REFERENCES.md` and did not reuse the earlier Sierpinski carpet proposal. |
| **3. Build an independent reference first:** How can I generate a small correct version before using PyTorch? | The AI suggested a NumPy/Python reference based on Pascal-triangle parity, plus an independent comparison with exact binomial coefficients. It also identified symmetry and the known count `3^n` in the first `2^n` rows as useful checks. | I kept the reference implementation separate from the PyTorch implementation. I ran `tests/verify_reference.py`, checked small cases, and visually inspected `outputs/reference_gasket.png`. This made the reference a correctness oracle rather than merely another rendering. |
| **4. Convert the calculation to PyTorch:** How can the pixel tests be evaluated in parallel without a Python loop over pixels? | The AI suggested creating row and column address tensors with shapes `(N, 1)` and `(1, N)`. PyTorch broadcasting then evaluates an `(N, N)` candidate grid using comparison and bitwise tensor operations. It also suggested keeping tensors on the selected device until the final Matplotlib conversion. | I reviewed the tensor shapes and the meaning of bitwise AND. I kept the core pixel calculation loop-free and compared its Boolean mask element-for-element with the independent NumPy result. The submitted demo explicitly uses `--device cpu`, although the code also supports MPS and CUDA when available. |
| **5. Decide whether a plausible image is enough:** How should I verify the PyTorch output? | The AI recommended automated checks for equality with the reference, known point counts, dtype, device preservation, invalid inputs, and preservation of points during the centred display mapping. | I did not accept the result just because it looked like a triangle. I ran all verification scripts locally. The 256-row PyTorch result contains 6,561 points, matching `3^8`, and agrees exactly with the reference. |
| **6. Add analysis beyond basic generation:** What analysis would show understanding of the fractal rather than only producing a picture? | The AI suggested box counting, comparison with the theoretical dimension `log(3)/log(2)`, an address-based colour interpretation, and a repeatable timing experiment. It warned that a single timing measurement would include startup and allocation noise. | I used aligned power-of-two box sizes and explained why this deterministic construction gives exact self-similar counts. The measured dimension was `1.584962501` with `R^2 = 1.0`. I used binary-address balance for the colour plot, rather than arbitrary decorative colours. For timing, I used three discarded warm-ups and nine measured repetitions at each size and reported the median as the main statistic. |
| **7. Reconsider Colab:** “The teacher said it can even run on CPU; remove the Colab part.” | The AI identified the Colab notebook, Colab instructions, bundle script, and CUDA-specific outputs that had become unnecessary. It proposed keeping the implementation device-capable while making the submitted evidence fully local and CPU-based. | I chose to remove the Colab-specific files because they made the workflow more complicated without supporting the final demo. I regenerated the figures and benchmark on my own Mac CPU and updated the documentation to match what was actually run. |
| **8. Audit the GitHub submission:** “Is my GitHub submission incomplete? Did I fail to submit the generated result images?” | The AI inspected the repository and found that generated outputs were excluded by the previous ignore rules. It proposed tracking a selected set of figures and CSV evidence while continuing to ignore temporary files. | I reviewed the selected outputs, updated the ignore rules, restored unrelated test files that should not have been removed, and committed four figures plus the raw benchmark and box-count data. The README now displays the figures so they are available during the demonstration. |
| **9. Prepare the AI declaration:** Create a statement that shows the thinking and revision process instead of implying that one AI prompt generated the project. | The AI drafted this structured declaration from the actual stages, including suggestions that were rejected or changed and the evidence used to validate retained work. | I reviewed this statement against my work and repository history before submission. I remain responsible for ensuring that it accurately describes my use of AI and for explaining the implementation during the demo. |

## Important modifications to AI-assisted suggestions

The following changes are especially important because they show where I made
decisions after reviewing the AI output:

1. **Remote execution was removed.** An early version included a Colab/CUDA
   workflow because I was uncertain whether remote GPU execution was expected.
   After the CPU clarification, I removed those files and used local CPU
   evidence.
2. **Correctness was checked independently.** I did not rely on the AI-generated
   PyTorch code or its image alone. I compared it with exact binomial
   coefficients, a separate NumPy implementation, known `3^n` counts, and
   automated tests.
3. **The benchmark method was strengthened.** A one-off runtime was rejected as
   insufficient. The final experiment uses warm-ups, repeated measurements,
   raw CSV records, and median/mean/standard-deviation summaries.
4. **The colour visualisation was made meaningful.** The colour represents the
   balance of 1-bits in the two non-overlapping binary addresses. It is tied to
   the construction rather than being an arbitrary colour map.
5. **Submission evidence was corrected.** When I realised that locally generated
   images were not tracked, I changed the repository rules and committed only
   the relevant final figures and data. I also checked that the README embeds
   them for the demo.

## Verification performed by me

I ran the following checks in my local `comp3710` environment:

```bash
python tests/verify_reference.py
python tests/verify_torch_gasket.py
python tests/verify_analysis.py
python tests/verify_benchmark.py
```

All four verification scripts passed. I also regenerated and visually reviewed:

- `outputs/reference_gasket.png`;
- `outputs/torch_gasket.png`;
- `outputs/gasket_analysis.png`; and
- `outputs/benchmark.png`.

The mathematical source, the course source, and the distinction between sourced
ideas and student/AI-assisted work are recorded in `REFERENCES.md`. Detailed
explanations of the tensor construction, display mapping, box counting, colour
meaning, and benchmark limitations are recorded in `PART3_NOTES.md`.

## Responsibility statement

AI assistance does not replace my responsibility for this submission. I chose
the final scope, ran the experiments, checked the results, removed material that
was no longer relevant, and decided what to commit. I understand that AI output
can be incomplete or incorrect, so I treated it as a source of suggestions to
be tested rather than as authoritative evidence. I am prepared to explain the
mathematical condition, PyTorch broadcasting, verification strategy, analysis,
and limitations of the benchmark during the demonstration.
