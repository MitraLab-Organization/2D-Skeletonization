# PHD Skeleton Detection - Final Results

## ✅ Current Configuration

**Parameters:** Code defaults (ro=5, ni=5, kappa=3, ps=0.95, pd=0.95, krad=4, kc=50)  
**Distance threshold:** 5 pixels (MATLAB default)  
**Input directory:** `tilesPMD/lkl/`

---

## 📊 Results Summary

### PHD Model (Code Defaults):
| Metric    | Value      |
|-----------|------------|
| Precision | **72.32%** |
| Recall    | **7.52%**  |
| F-Score   | **13.62%** |
| Dice      | **13.62%** |
| IoU       | **7.31%**  |

### Baseline (bwskel):
| Metric    | Value      |
|-----------|------------|
| Precision | **74.94%** |
| Recall    | **51.42%** |
| F-Score   | **60.99%** |
| Dice      | **60.99%** |
| IoU       | **43.88%** |

### Interpretation:
- ✅ **PHD is precise** (72% precision) - what it detects is mostly correct
- ⚠️ **PHD has low recall** (7.5%) - it misses ~92% of neurons
- 🏆 **Baseline (bwskel) is better orall** - good balance of precision (75%) and recall (51%)

---

## 🚀 Complete Workflow

### 1. Run PHD Detection on Images

```bash
cd ~/Projects
java -Xmx4g -jar /usr/share/java/ij.jar -ijpath ~/.imagej/plugins/ \
  -batch ~/Projects/phd/ij-macro/run_phd_preprocess.ijm \
  ~/Projects/Data/WholeBrainProject/tilesPMD/tilesPMD/lkl/
```

**Output:** SWC skeleton files saved in subdirectories within `lkl/`

### 2. Convert SWC to Binary Images

```bash
cd ~/Projects/Data/WholeBrainProject/tilesPMD
python3 convert_swc_to_images.py \
  --phd_dir tilesPMD/lkl \
  --output_dir tilesPMD/det \
  --reference_dir tilesPMD/img
```

**Output:** Binary skeleton images in `tilesPMD/det/`

### 3. Compare Against Baseline and Ground Truth

```bash
cd ~/Projects/Data/WholeBrainProject/tilesPMD
python3 run_comparison_tilesPMD.py
```

**Output:** Metrics and visualizations in `tilesPMD/comparison_results/`

---

## 📁 Directory Structure

```
tilesPMD/
├── lkl/                         # Input images for PHD (CORRECT INPUT DIR)
├── img/                         # Reference images (same as lkl/)
├── det/                         # PHD predictions (converted from SWC)
├── GT/                          # Ground truth skeletons
├── skel_lkl/                    # bwskel baseline skeletons
├── comparison_results/          # Final comparison metrics
│   ├── detailed_results_pred.csv
│   ├── detailed_results_baseline.csv
│   ├── summary.json
│   ├── overlays_pred/          # PHD visualization
│   └── overlays_baseline/      # bwskel visualization
├── skeleton_comparison.py       # Comparison engine
├── convert_swc_to_images.py    # SWC → Binary converter
└── run_comparison_tilesPMD.py  # Ready-to-run comparison
```

---

## 🎨 Overlay Color Coding

Visual comparisons show:
- **Cyan**: True Positives (correctly detected skeleton pixels)
- **Magenta**: False Negatives (GT pixels that were missed)
- **Yellow**: False Positives (incorrect skeleton predictions)

---

## 🔧 Scripts

### `skeleton_comparison.py`
Main comparison engine using:
- KD-tree nearest neighbor matching (5-pixel threshold)
- Computes TP, FP, FN
- Calculates Precision, Recall, F-Score, Dice, IoU
- Generates overlay visualizations

### `convert_swc_to_images.py`
Converts PHD output (SWC format) to binary TIFF images for comparison.

### `run_comparison_tilesPMD.py`
Pre-configured script to run complete comparison with correct paths.

---

## 📈 Key Insights

1. **Code defaults outperform paper defaults** significantly
   - Code: Precision 72%, Recall 7.5%
   - Paper defaults: Precision 11%, Recall 0.2% ❌

2. **Distance threshold (5px) is appropriate**
   - Tested 3, 5, and 7 pixels
   - Results vary by < 1% across all thresholds
   - 5 pixels is MATLAB default and well-established

3. **Main challenge: Low recall**
   - PHD is conservative and misses many neurons
   - This is a coverage issue, not an accuracy issue
   - Baseline bwskel has better recall (51% vs 7.5%)

4. **Baseline (bwskel) is currently superior**
   - Better F-score (61% vs 14%)
   - Better balance of precision and recall
   - May be the better production method

---

## 💡 Future Work

To improve PHD recall while maintaining precision:
1. Try intermediate parameter values (ro=7, ni=7)
2. Adjust detection thresholds (lower th, pd, ps slightly)
3. Experiment with preprocessing (contrast enhancement)
4. Consider ensemble methods (PHD + bwskel)

---

## ✅ Files Generated

All results are in `outputs/pmd/` relative to the project root.

**Key files:**
- `SUMMARY.md` - This file
- `comparison_results/summary.json` - Metric summary
- `comparison_results/overlays_*/` - Visual comparisons
