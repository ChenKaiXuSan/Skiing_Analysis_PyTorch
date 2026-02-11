# Angle Computation Method

This document describes how joint angles are computed in `main.py` for the skiing athlete.

## Input

- The input file is a NumPy array saved as `.npy` with shape `(T, J, 3)`.
- `T` is the number of frames.
- `J` is the number of joints, ordered by `TARGET_IDS`.
- Each joint is a 3D point `(x, y, z)` in the same coordinate system as the keypoints.

## Joint Indexing

The mapping `TARGET_IDS` is built from `UNITY_MHR70_MAPPING` and defines the index order in the array.
`ID_TO_INDEX` maps each joint ID to its array index.

Example entries:

- `9`: Thigh_L
- `11`: calf_l
- `13`: Foot_L
- `69`: neck_01

## Angle Definition

Each angle is defined by three joint IDs `(A, B, C)` and is the angle at joint `B` formed by the segments `BA` and `BC`.

The default definitions are:

- `knee_l`: `(9, 11, 13)`
- `knee_r`: `(10, 12, 14)`
- `elbow_l`: `(5, 7, 62)`
- `elbow_r`: `(6, 8, 41)`
- `shoulder_l`: `(69, 5, 7)`
- `shoulder_r`: `(69, 6, 8)`
- `hip_l`: `(69, 9, 11)`
- `hip_r`: `(69, 10, 12)`

You can change these tuples in `ANGLE_DEFS` to compute different joint angles.

## Angle Formula

For a single frame and a single angle `(A, B, C)`:

- `BA = A - B`
- `BC = C - B`

The angle is:

$$
\theta = \arccos \left( \frac{BA \cdot BC}{\|BA\| \, \|BC\|} \right)
$$

The result is converted from radians to degrees.

## Missing or Invalid Data

- If any of the three joints has a non-finite value (`NaN` or `Inf`), the angle for that frame is `NaN`.
- If either vector length is zero, the angle is `NaN`.

## Output

- A CSV file with one row per frame and one column per angle.
- A PNG plot with one subplot per angle showing the time series.

## Notes

- The angle computation is purely geometric; no temporal smoothing is applied here.
- If you need ankle angles, you must include toe or foot tip joints in `TARGET_IDS` and define `(calf, ankle, toe)` tuples.
