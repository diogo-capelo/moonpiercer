# TODO — MOONPIERCER Pipeline

## Priority: Missing-shape penalty (Change 4)

Shape-unreliable craters currently receive `T_ellipticity = T_orientation = T_position = 1.0`,
inflating their scores relative to shape-reliable craters. With ~216k shape-unreliable craters
in a 3D attribute space (radius, NLS, RCR), the null model always finds near-perfect matches
(best NN score ~0.9999999999), making significance testing impossible for this population.

**Goal**: Introduce a multiplicative penalty (or reduced score ceiling) for pairs where one
or both craters lack reliable shape, so that shape-reliable pairs — which have 6 active
scoring dimensions — are not drowned out.

**Considerations**:
- The penalty must be calibrated so that a genuine PBH pair with unreliable shape can still
  rank highly, while random attribute matches among 216k craters cannot.
- One approach: set `T_ellipticity = T_orientation = T_position = c` (e.g. c = 0.5) when
  shape is unreliable, instead of 1.0.
- Another approach: apply a single `T_shape_penalty` factor based on the number of missing
  geometric constraints.
- Assess impact on null score distribution before finalising.

---

## Future: LOLA depth as a 4th shape-independent dimension

The current shape-independent scoring uses 3 dimensions (radius, NLS, RCR). Adding LOLA-derived
crater depth as a 4th dimension would improve discrimination for shape-unreliable craters.

- LOLA 128 ppd DTM is already fetched by the pipeline (`use_lola_topography` config).
- Depth could be measured as the difference between rim elevation and floor elevation
  within the detected crater footprint.
- Cross-observation noise calibration would be needed (same approach as NLS/RCR: measure
  |depth_a - depth_b| for matched craters across observations).
- Expected benefit: breaks the degeneracy in 3D attribute space that makes all large
  shape-unreliable catalogues indistinguishable from random.

---

## Future: Crater intensity profile matching

Instead of scalar summary statistics (NLS, RCR), compare the full radial intensity profile
of paired craters. Two craters formed by the same PBH should have similar radial brightness
profiles (normalised for illumination differences).

- Extract azimuthally-averaged radial profiles from NAC chips.
- Use cross-correlation or chi-squared distance as a scoring term.
- This adds many effective dimensions of comparison, dramatically increasing discrimination
  even for shape-unreliable craters.
- Requires careful normalisation for solar incidence angle differences between observations.
