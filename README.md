# Sudoku Detector & Solver

Detects a Sudoku puzzle from a static image using OpenCV, recognizes digits with a CNN trained on MNIST, and solves the puzzle using backtracking — then overlays the solution on the original image.

## Overview

This project builds a pipeline that:
1. Detects the Sudoku grid in an image using **OpenCV** (contour detection + perspective warp)
2. Extracts each of the 81 cells and classifies digits with a **PyTorch CNN**
3. Solves the puzzle using a **backtracking algorithm**
4. Overlays the solution back onto the original image

## Workflow

| Step | Where |
|------|-------|
| Grid detection & inference | Local (Mac, CPU) |
| CNN training on MNIST | Google Colab (GPU) → download `models/digit_cnn.pt` |

To train the model, open `notebook/sudoku_solver.ipynb` in Google Colab, run the training section, and download the saved weights into `models/`.

## Tech Stack

| Tool | Role |
|------|------|
| OpenCV | Grid detection, perspective warp, cell extraction |
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
│   ├── raw/          # Input Sudoku images
│   └── processed/    # Extracted cell images (git-ignored)
├── models/
│   └── digit_cnn.pt  # Trained CNN weights (download from Colab)
├── notebook/
│   └── sudoku_solver.ipynb
└── requirements.txt
```

## Results

_To be filled in after training._

## License

MIT
