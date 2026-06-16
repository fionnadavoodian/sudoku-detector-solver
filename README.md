# Sudoku Detector & Solver

Detects a Sudoku puzzle from a photo, reads the digits with a CNN trained on MNIST, solves it with backtracking, and overlays the solution on the original image.

**Pipeline:** Photo → Resize → Grid Detection → Perspective Warp → Cell Extraction → Digit CNN → Solver → Overlay

## Current Status

| Section | Status |
|---------|--------|
| 1. Imports & Setup | ✅ Done |
| 2. Load & Resize Image | ✅ Done |
| 3. Grid Detection | ✅ Done |
| 4. Perspective Warp | ✅ Done |
| 5. Cell Extraction | ✅ Done |
| 6. CNN Digit Classifier | ✅ Done |
| 7. Read the Puzzle | ✅ Done (accuracy WIP) |
| 8. Backtracking Solver | ✅ Done |
| 9. Solution Overlay | 🚧 In progress |
| 10. End-to-End Pipeline | 🚧 In progress |

## How It Works

### Grid Detection (Section 3)
1. Resize input image to **900×900** for reliable processing
2. Convert to grayscale
3. **Adaptive threshold** (`MEAN_C`, block=25, C=2) → binary image
4. **Morphology** open+close (3×3 kernel) to clean noise
5. Separate **horizontal** (1×40 kernel) and **vertical** (40×1 kernel) lines
6. Combine lines → find contours → `approxPolyDP` → 4 grid corners

### Perspective Warp (Section 4)
Order the 4 corners (TL, TR, BR, BL) and use `cv2.getPerspectiveTransform` + `cv2.warpPerspective` to flatten the grid into a clean 900×900 top-down square.

### Cell Extraction (Section 5)
Split the warped grid into 81 cells and resize each to **28×28** for the CNN.

### CNN Digit Classifier (Section 6)
Small CNN trained on **MNIST** (5 epochs, Adam) — runs on CPU (no GPU needed):
```
Input (1×28×28)
→ Conv2d(1, 32, 3) + ReLU + MaxPool
→ Conv2d(32, 64, 3) + ReLU + MaxPool
→ Flatten → Linear(1600, 128) + ReLU + Dropout
→ Linear(128, 10)   # 0 = empty cell, 1–9 = digit
```
Weights saved as `notebook/digit_cnn.pt`.

### Backtracking Solver (Section 8)
Recursive backtracking — tries digits 1–9 in each empty cell, validates against row/column/3×3 box constraints, backtracks on conflict. Solves most puzzles in milliseconds.

## Tech Stack

| Tool | Role |
|------|------|
| OpenCV | Grid detection, morphology, perspective warp, cell extraction |
| PyTorch | CNN digit classifier |
| NumPy | Array operations |
| Matplotlib | Visualization |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebook/sudoku_solver.ipynb
```

## Project Structure

```
sudoku-detector-solver/
├── data/
│   ├── raw/              # Input Sudoku images
│   └── MNIST/            # Auto-downloaded during training
├── notebook/
│   ├── sudoku_solver.ipynb
│   └── digit_cnn.pt      # Trained CNN weights
└── requirements.txt
```

## License

MIT
