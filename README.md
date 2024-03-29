# Weighted least squares for pose estimation

Python implementation of closed-form least squares optimization methods for 3D pose estimation.
- Closed-form weighted least squares
    - Formulation from Malis (2023) [1]
- Umeyama algorithm [2]

# Usage

```
usage: lsq.py [-h] [--algorithm {umeyama,malis}] [--num-points NUM_POINTS] [--num-outliers NUM_OUTLIERS]

optional arguments:
  -h, --help            show this help message and exit
  --algorithm {umeyama,malis}
                        LSQ algorithm to use. (default: malis)
  --num-points NUM_POINTS
                        The number of points. (default: 10)
  --num-outliers NUM_OUTLIERS
                        The number of outliers. (default: 0)
```

# References

[1] E. Malis, "Complete Closed-Form and Accurate Solution to Pose Estimation From 3D Correspondences", IEEE Robotics and Automation Letters, 2023 [PDF](https://hal.science/hal-03957104/document)

[2] S. Umeyama, "Least-squares estimation of transformation parameters between two point patterns", IEEE Transactions on Pattern Analysis and Machine Intelligence, 1991 [PDF](https://web.stanford.edu/class/cs273/refs/umeyama.pdf)
