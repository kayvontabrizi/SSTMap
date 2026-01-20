## imports

# custom
import numpy


## methods


def rotate_check(matrix: numpy.ndarray):
    determinant = numpy.linalg.det(matrix)
    if not (0.99 < determinant < 1.01):
        raise Warning(
            "Warning: Determinant of rotation matrix is %s. Should be close to +1.0."
            % determinant
        )


def do_rotation(
    coordinates: numpy.ndarray,
    origin: numpy.ndarray,
    rotation_matrix: numpy.ndarray,
) -> numpy.ndarray:
    return (coordinates - origin).dot(rotation_matrix) + origin
