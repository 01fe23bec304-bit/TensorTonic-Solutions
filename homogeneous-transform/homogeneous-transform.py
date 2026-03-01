import numpy as np

def apply_homogeneous_transform(T, points):
    T = np.array(T, dtype=float)
    points = np.array(points, dtype=float)

    single_point = False
    if points.ndim == 1:
        single_point = True
        points = points.reshape(1, -1)

    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack((points, ones))

    # Apply transform
    transformed_h = points_h @ T.T

    # Convert back to 3D
    transformed = transformed_h[:, :3]

    if single_point:
        return transformed[0].tolist()

    return transformed.tolist()