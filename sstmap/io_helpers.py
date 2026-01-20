# encoding: utf-8

## imports

# standard
import typing

# custom
import numpy


## methods


def are_you_numpy(value: typing.Any) -> bool:
    return type(value).__module__ == numpy.__name__


def make_grid(
    arrays: tuple[numpy.ndarray, ...],
    out: typing.Optional[numpy.ndarray] = None,
) -> numpy.ndarray:
    arrays_list = [numpy.asarray(x) for x in arrays]
    dtype = arrays_list[0].dtype
    total_elements = numpy.prod([x.size for x in arrays_list])

    if out is None:
        out = numpy.zeros([total_elements, len(arrays_list)], dtype=dtype)

    multiplier = total_elements // arrays_list[0].size
    out[:, 0] = numpy.repeat(arrays_list[0], multiplier)

    if arrays_list[1:]:
        make_grid(tuple(arrays_list[1:]), out=out[0:multiplier, 1:])
        for index in range(1, arrays_list[0].size):
            out[index * multiplier : (index + 1) * multiplier, 1:] = out[
                0:multiplier, 1:
            ]

    return out


def bounding_box_frac(
    fractional_structure: numpy.ndarray,
    delta: numpy.ndarray = numpy.ones(3),
    buffer_size: float = 0.0,
    verbose: bool = False,
) -> numpy.ndarray:
    bounding_min = numpy.array(
        [
            numpy.min(fractional_structure[:, 0]),
            numpy.min(fractional_structure[:, 1]),
            numpy.min(fractional_structure[:, 2]),
        ],
        dtype=int,
    )

    bounding_max = numpy.array(
        [
            numpy.max(fractional_structure[:, 0]),
            numpy.max(fractional_structure[:, 1]),
            numpy.max(fractional_structure[:, 2]),
        ],
        dtype=int,
    )

    bounding_min -= int(numpy.round(buffer_size))
    bounding_max += int(numpy.round(buffer_size))

    if verbose:
        print("Bounding min. ", bounding_min)
        print("Bounding max. ", bounding_max)
        print(numpy.arange(bounding_min[2], bounding_max[2] + 1, delta[2], dtype=int))

    return make_grid(
        (
            numpy.arange(bounding_min[0], bounding_max[0] + 1, delta[0], dtype=int),
            numpy.arange(bounding_min[1], bounding_max[1] + 1, delta[1], dtype=int),
            numpy.arange(bounding_min[2], bounding_max[2] + 1, delta[2], dtype=int),
        )
    )
