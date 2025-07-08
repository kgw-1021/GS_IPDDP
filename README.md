# GS_IPDDP

**Trajectory Optimization for Quadrotors Using Gaussian Splatting-based Density Fields with IPDDP**

This repository implements a real-time trajectory optimization framework for aerial robots (e.g., quadrotors), utilizing **pre-trained Gaussian Splatting (GS)** maps as a 3D environmental representation. The system transforms GS into a **differentiable voxelized density field**, allowing integration with **Interior Point Differential Dynamic Programming (IPDDP)** for smooth and collision-free path planning.

---

## 🔍 Overview

This project is built on the following insights:

- Gaussian Splatting (GS) provides a compact, high-fidelity 3D representation using anisotropic Gaussians.
- The **density field** derived from GS can be queried and differentiated to inform collision-aware trajectory optimization.
- We formulate a cost function that penalizes high-density regions (i.e., obstacles), and optimize it under drone dynamics via **IPDDP**.

---



