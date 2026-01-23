# Baselines

This folder contains baseline skeletonization methods for comparison.

## Running Baselines Manually

### NeuTube
```bash
# Run on likelihood images
python Baselines/neutube/batch_neutube.py -i data/pmd/lkl -o outputs/pmd/neutube_swc

# Convert SWC to TIF
python Baselines/neutube/batch_convert_swc.py --swc_dir outputs/pmd/neutube_swc --lkl_dir data/pmd/lkl --out_dir outputs/pmd/neutube
```

### VESS (Vesselness Filter)
```bash
python Baselines/vess/run_vess.py --dataset pmd
```
*Requires Fiji installed at `/opt/Fiji.app/`*

### PHD (Probability Hypothesis Density)
```bash
python Baselines/phd/run_phd.py --dataset pmd
```
*Requires Fiji installed at `/opt/Fiji.app/`*

### Differentiable Skeletonization
```bash
python Baselines/skeletonization-for-gradient-based-optimization/batch_process.py --input_folder data/pmd/lkl --output_folder outputs/pmd/diffskel
```

### bwskel (MATLAB)
Pre-computed outputs are included in `outputs/pmd/bwskel/` and `outputs/stp/bwskel/`.

To run manually (requires MATLAB):
```matlab
run('Baselines/bwskel/run_bwskel.m')
```

## Directory Structure
```
Baselines/
├── neutube/           # NeuTube neuron tracing
├── vess/              # Vesselness filter + skeleton
├── phd/               # PHD filter (ImageJ plugin)
├── bwskel/            # MATLAB bwskel (pre-computed)
├── skeletonization-for-gradient-based-optimization/  # Differentiable skeleton
└── imagej_plugins/    # ImageJ/Fiji plugins
```
