# Ion Track & 1D/2D Ion Occupancy Analysis for Channel Simulations

This repository contains a Python script that performs quantitative ion tracking and occupancy analysis for molecular dynamics (MD) simulations of ion channels using **MDAnalysis**.

It computes:

- **Ion trajectories along the pore axis (z-axis)**
- **1D ion occupancy** along the pore axis
- **2D radial–axial ion density maps**
- Snapshot-overlay visualization of 2D ion occupancy
- Publication-ready figures (PDF format)

The workflow is designed for potassium (K⁺) channel simulations but can be adapted to other ions (Na⁺, Cl⁻, etc.).

---

## What the script does

1. Aligns the channel to a reference structure (via a user-defined atom selection)
2. Defines a cylindrical analysis region around the channel center of mass (COM)
3. Tracks ions inside that cylinder over time
4. Computes:
   - Per-ion **z-position vs time** (Ion Track)
   - **1D axial ion occupancy**
   - **2D radial–axial ion density** (normalized by cylindrical shell volume)
5. Writes high-resolution figures (PDF):
   - `IonTrack+.pdf`
   - `1DIO+.pdf`
   - `2DIO+.pdf`

---

## Dependencies

- Python ≥ 3.7  
- [MDAnalysis](https://www.mdanalysis.org/)
- NumPy
- SciPy
- Matplotlib

---

## User-defined parameters to edit

Open the script and adapt these sections to your system:

- `Path` → trajectory directory  
- `skip` → frame skipping for performance  
- `dt` → integration timestep (ps)  
- `outputstep` → trajectory output stride  
- Cylinder dimensions:
  - `height`
  - `depth`
  - `radius`
- Ion selection (e.g. `traj.select_atoms('name K', updating=True)`)
- Alignment atom selection (channel-specific indices / atom names)
- Optional: snapshot image path and pixel scaling parameters used for 2D overlay plots

---

## Method notes

### Alignment
Each analyzed frame is translated so that the channel backbone selection is recentered to the reference center of mass. This stabilizes the pore axis for ion tracking.

### Ion tracking and occupancy
For each analyzed frame:
- Ions inside the cylindrical region are selected (updated every frame)
- The ion **z-position** (relative to the reference COM) is stored in an ion track array
- 1D and 2D occupancy histograms are updated

### 2D normalization (radial shells)
Radial bins represent cylindrical shells; outer shells cover larger volumes than inner shells.  
To report a density-like quantity, each radial bin is normalized by its shell volume:

\[
V_{shell} = \pi \cdot h \cdot (r_{outer}^2 - r_{inner}^2)
\]

Resulting units: ions per Å³ (summed over the trajectory).

---

## Outputs

The script saves plots into the trajectory `Path` directory:

- **Ion track**: time vs z-position for each ion (`IonTrack+.pdf`)
- **1D occupancy** along the pore axis (`1DIO+.pdf`)
- **2D occupancy/density** (optionally overlaid on a structural snapshot) (`2DIO+.pdf`)

---

## License

Add a license file if you plan to share this publicly (e.g., MIT, BSD-3, GPL-3).
