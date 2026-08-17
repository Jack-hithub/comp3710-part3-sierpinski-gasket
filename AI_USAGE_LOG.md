# AI usage and revision log

This log records AI assistance while it happens. It should preserve mistakes,
questions, rejected ideas, and corrections rather than inventing a clean
process after completion.

The model cannot provide hidden private chain-of-thought. The entries instead
record the useful output and a concise, inspectable reasoning summary that the
student can verify against the code and sources.

## 2026-08-18 - Replacing the original proposal

### Exact user prompts

> 老师允许用这个Sierpiński carpet吗

> 教材《Fractals for the Classroom》中的分形；Shakes 发现的分形；Wikipedia 列表里的其他分形需要 teaching staff 批准；这些都是什么，我去找来，然后你判断吧

> 我下载下来了，你自己去看吧

### AI assistance and output

- Located and reviewed the complete textbook sections named by the lab sheet:
  2.10, 3.6, 4.6, and 6.6.
- Identified that Section 2.10 contains the **Sierpinski gasket**, not the
  previously proposed **Sierpinski carpet**.
- Compared the four named textbook choices for implementation difficulty,
  GPU suitability, analysis opportunities, and approval risk.
- Recommended the gasket because it is explicitly named by the textbook
  example, differs clearly from the Part 2 escape-time fractals, and permits
  independent per-pixel binary tests.

### Review decision and correction

- The carpet proposal was not treated as approved: as a Wikipedia-list choice,
  it would require teaching-staff approval.
- The student accepted switching to the textbook's Sierpinski gasket and
  deleted the old carpet directory.

## 2026-08-18 - Clean project restart

### Exact user prompt

> 可以的，之前那个文件夹我直接删了，你重新开始吧

### AI assistance and reasoning summary

- Verified that both the deleted carpet directory and the new gasket directory
  were absent before creating anything.
- Found that macOS system Python 3.9 does not have PyTorch, then located and
  verified the existing conda environment instead of installing duplicates.
- Confirmed Python 3.11.15, PyTorch 2.13.0, NumPy 2.4.6, and Matplotlib 3.11.1
  in the `comp3710` environment.
- Created a new project structure, staged plan, source record, environment
  file, and this log. No GitHub repository or remote was created.

### Student review still required

- Understand why the gasket is not the same fractal as the carpet.
- Review the small reference algorithm and its checks before the PyTorch/GPU
  implementation begins.

## 2026-08-18 - Independent reference stage

### AI assistance and output

- Restated the textbook's Pascal-triangle parity condition in original NumPy
  code rather than copying its BASIC program.
- Kept the reference independent of PyTorch so it can later detect errors in
  the GPU implementation.
- Added a centred raster mapping solely for display; this mapping does not
  change which Pascal entries are odd.
- Added exact-integer checks using `math.comb`, known first rows, row symmetry,
  the `3^n` count for the first `2^n` rows, and preservation of points during
  raster mapping.

### Validation and observed output

- Python compilation completed successfully.
- All independent reference checks passed for the tested sizes.
- For 128 rows, the program found 2,187 odd entries, agreeing with
  `3^7 = 2,187`, and produced a `128 x 255` raster.
- `outputs/reference_gasket.png` was visually inspected. It has a centred
  triangular outline and the expected recursively repeated triangular gaps.
- Matplotlib reported that the managed execution environment could not write
  to its normal font cache. It used a temporary cache; this did not affect the
  calculation or saved image.

### Review boundary

- No PyTorch conversion, GPU benchmark, dimension estimator, or final colour
  design was added in this stage.
- The student should first be able to explain the reference condition
  `column & (row - column) == 0` at a high level.
