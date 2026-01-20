## imports

# standard
import os
import gzip
import typing
import collections

# custom
import numpy
import string

# local
from sstmap import io_spatial
from sstmap import io_helpers


## classes


class field(object):
    def __init__(
        self,
        Bins: numpy.ndarray,
        Frac2Real: typing.Optional[numpy.ndarray] = None,
        Delta: typing.Optional[numpy.ndarray] = None,
        Origin: typing.Optional[numpy.ndarray] = None,
        Center: typing.Optional[numpy.ndarray] = None,
    ):
        if Frac2Real is None and Delta is None:
            raise ValueError("Must provide Frac2Real or Delta.")

        if Frac2Real is not None and Delta is not None:
            raise ValueError("Must provide either Frac2Real or Delta.")

        if Frac2Real is None:
            self.delta = Delta
            self.frac2real = numpy.eye(3, 3) * self.delta
        else:
            self.frac2real = Frac2Real
            self.delta = numpy.linalg.norm(self.frac2real, axis=0)

        self.real2frac = numpy.linalg.inv(self.frac2real)
        self.bins = Bins
        self.rotation_matrix = numpy.eye(3, 3)
        self.translation_vector = numpy.zeros(3)

        if Origin is None and Center is None:
            raise ValueError("Must provide origin or Center.")

        if Origin is not None and Center is not None:
            raise ValueError("Must provide either origin or center.")

        if Center is None:
            self.origin = Origin
            self.center = self.get_real(self.bins / 2)
        else:
            self.center = Center
            self.origin = numpy.zeros(3)
            self.origin = self.center - self.get_real(self.bins / 2)

        self.dim = numpy.array(
            [
                numpy.linalg.norm(
                    self.get_real([self.bins[0], 0.0, 0.0]) - self.origin
                ),
                numpy.linalg.norm(
                    self.get_real([0.0, self.bins[1], 0.0]) - self.origin
                ),
                numpy.linalg.norm(
                    self.get_real([0.0, 0.0, self.bins[2]]) - self.origin
                ),
            ]
        )

    def translate(self, vector: numpy.ndarray = numpy.zeros(3)):
        self.translation_vector += vector

    def rotate(self, matrix: numpy.ndarray = numpy.eye(3, 3)):
        io_spatial.rotate_check(matrix)
        self.rotation_matrix = matrix.dot(self.rotation_matrix)

    def translate_global(self, vector: numpy.ndarray = numpy.zeros(3)):
        self.origin += vector

    def rotate_global(
        self,
        reference_point: numpy.ndarray = numpy.zeros(3),
        matrix: numpy.ndarray = numpy.eye(3, 3),
    ):
        io_spatial.rotate_check(matrix)
        self.origin = io_spatial.do_rotation(self.origin, reference_point, matrix)
        self.rotate(matrix)
        self.translation_vector = io_spatial.do_rotation(
            self.translation_vector, numpy.zeros(3), matrix
        )

    def get_nice_frac2real(self) -> numpy.ndarray:
        return self.rotation_matrix.dot(self.frac2real)

    def get_nice_real2frac(self) -> numpy.ndarray:
        return numpy.linalg.inv(self.get_nice_frac2real())

    def get_voxel_volume(self) -> float:
        return numpy.absolute(
            numpy.cross(self.frac2real[:, 0], self.frac2real[:, 1]).dot(
                self.frac2real[:, 2]
            )
        )

    def get_frac(self, real_array: numpy.ndarray) -> numpy.ndarray:
        initial_reals = io_spatial.do_rotation(
            real_array,
            self.origin + self.translation_vector,
            numpy.linalg.inv(self.rotation_matrix),
        )
        initial_reals -= self.origin + self.translation_vector
        return initial_reals.dot(self.real2frac)

    def get_real(self, frac_array: numpy.ndarray) -> numpy.ndarray:
        reals = numpy.array(frac_array).dot(self.frac2real)
        return (
            io_spatial.do_rotation(reals, numpy.zeros(3), self.rotation_matrix)
            + self.origin
            + self.translation_vector
        )

    def get_centers(self) -> numpy.ndarray:
        return self.get_real(
            io_helpers.make_grid(
                (
                    numpy.arange(self.bins[0]),
                    numpy.arange(self.bins[1]),
                    numpy.arange(self.bins[2]),
                )
            )
        )

    def get_centers_real(self) -> numpy.ndarray:
        return self.get_centers()

    def get_centers_frac(self) -> numpy.ndarray:
        return io_helpers.make_grid(
            (
                numpy.arange(self.bins[0]),
                numpy.arange(self.bins[1]),
                numpy.arange(self.bins[2]),
            )
        )


class gist(field):
    def __init__(
        self,
        Bins: numpy.ndarray,
        Frac2Real: typing.Optional[numpy.ndarray] = None,
        Delta: typing.Optional[numpy.ndarray] = None,
        Origin: typing.Optional[numpy.ndarray] = None,
        Center: typing.Optional[numpy.ndarray] = None,
        gist17: bool = False,
    ):
        field.__init__(self, Bins, Frac2Real, Delta, Origin, Center)

        if type(gist17) != bool:
            raise IOError(
                "gist17 must be of type bool but is of type %s" % type(gist17)
            )

        self.gist17 = gist17
        self.bulk_NN = 5.098076
        self.ref_ene = 11.063656
        self.ref_rho = 0.0332

        self.Pop = numpy.zeros([self.bins[0], self.bins[1], self.bins[2]], dtype=float)
        self.gO = numpy.zeros([self.bins[0], self.bins[1], self.bins[2]], dtype=float)
        self.gH = numpy.zeros([self.bins[0], self.bins[1], self.bins[2]], dtype=float)
        self.dTStrans_dens = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self.dTStrans_norm = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self.dTSorient_dens = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self.dTSorient_norm = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self.dTSsix_dens = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self.dTSsix_norm = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self.Esw_dens = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self.Esw_norm = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self.Eww_dens = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self.Eww_norm_unref = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self.Dipole_x_dens = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self.Dipole_y_dens = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self.Dipole_z_dens = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self.Dipole_dens = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self.Neighbor_dens = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self.Neighbor_norm = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self.Order_norm = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )

        self._tmp_Pop = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self._tmp_gO = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self._tmp_gH = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self._tmp_dTStrans_dens = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self._tmp_dTStrans_norm = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self._tmp_dTSorient_dens = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self._tmp_dTSorient_norm = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self._tmp_dTSsix_dens = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self._tmp_dTSsix_norm = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self._tmp_Esw_norm = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self._tmp_Esw_dens = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self._tmp_Eww_dens = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self._tmp_Eww_norm_unref = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self._tmp_Dipole_x_dens = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self._tmp_Dipole_y_dens = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self._tmp_Dipole_z_dens = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self._tmp_Dipole_dens = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self._tmp_Neighbor_dens = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self._tmp_Neighbor_norm = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )
        self._tmp_Order_norm = numpy.zeros(
            [self.bins[0], self.bins[1], self.bins[2]], dtype=float
        )

        self._update()

    def _update(self):
        self.Pop = self._tmp_Pop
        self.gO = self._tmp_gO
        self.gH = self._tmp_gH
        self.dTStrans_dens = self._tmp_dTStrans_dens
        self.dTStrans_norm = self._tmp_dTStrans_norm
        self.dTSorient_dens = self._tmp_dTSorient_dens
        self.dTSorient_norm = self._tmp_dTSorient_norm
        self.dTSsix_dens = self._tmp_dTSsix_dens
        self.dTSsix_norm = self._tmp_dTSsix_norm
        self.Esw_dens = self._tmp_Esw_dens
        self.Esw_norm = self._tmp_Esw_norm
        self.Eww_dens = self._tmp_Eww_dens
        self.Eww_norm_unref = self._tmp_Eww_norm_unref
        self.Dipole_x_dens = self._tmp_Dipole_x_dens
        self.Dipole_y_dens = self._tmp_Dipole_y_dens
        self.Dipole_z_dens = self._tmp_Dipole_z_dens
        self.Dipole_dens = self._tmp_Dipole_dens
        self.Neighbor_dens = self._tmp_Neighbor_dens
        self.Neighbor_norm = self._tmp_Neighbor_norm
        self.Order_norm = self._tmp_Order_norm

    def cut_round_center(self, bins: numpy.ndarray):
        if bins.shape[0] != 3:
            print("Target bins array must habe shape (3,)")
        elif self.bins[0] < bins[0] or self.bins[1] < bins[1] or self.bins[2] < bins[2]:
            print("Target bins array must be smaller than original one.")
        else:
            cut = (self.bins - bins) / 2
            cut_bins = numpy.array(
                [[cut[0], cut[0]], [cut[1], cut[1]], [cut[2], cut[2]]]
            )
            self.cut(cut_bins)

    def cut(self, cut_bins: numpy.ndarray):
        self.bins[0] = self.bins[0] - cut_bins[0, 0] - cut_bins[0, 1]
        self.bins[1] = self.bins[1] - cut_bins[1, 0] - cut_bins[1, 1]
        self.bins[2] = self.bins[2] - cut_bins[2, 0] - cut_bins[2, 1]

        self.dim = self.bins * self.delta
        self.n = numpy.copy(self.bins)

        self.origin = numpy.zeros(3)
        self.origin = self.center - self.get_real(self.bins / 2)

        self._tmp_Pop = numpy.array(
            numpy.zeros([self.bins[0], self.bins[1], self.bins[2]]), dtype=float
        )
        self._tmp_gO = numpy.array(
            numpy.zeros([self.bins[0], self.bins[1], self.bins[2]]), dtype=float
        )
        self._tmp_gH = numpy.array(
            numpy.zeros([self.bins[0], self.bins[1], self.bins[2]]), dtype=float
        )
        self._tmp_dTStrans_dens = numpy.array(
            numpy.zeros([self.bins[0], self.bins[1], self.bins[2]]), dtype=float
        )
        self._tmp_dTStrans_norm = numpy.array(
            numpy.zeros([self.bins[0], self.bins[1], self.bins[2]]), dtype=float
        )
        self._tmp_dTSorient_dens = numpy.array(
            numpy.zeros([self.bins[0], self.bins[1], self.bins[2]]), dtype=float
        )
        self._tmp_dTSorient_norm = numpy.array(
            numpy.zeros([self.bins[0], self.bins[1], self.bins[2]]), dtype=float
        )
        self._tmp_dTSsix_dens = numpy.array(
            numpy.zeros([self.bins[0], self.bins[1], self.bins[2]]), dtype=float
        )
        self._tmp_dTSsix_norm = numpy.array(
            numpy.zeros([self.bins[0], self.bins[1], self.bins[2]]), dtype=float
        )
        self._tmp_Esw_norm = numpy.array(
            numpy.zeros([self.bins[0], self.bins[1], self.bins[2]]), dtype=float
        )
        self._tmp_Esw_dens = numpy.array(
            numpy.zeros([self.bins[0], self.bins[1], self.bins[2]]), dtype=float
        )
        self._tmp_Eww_dens = numpy.array(
            numpy.zeros([self.bins[0], self.bins[1], self.bins[2]]), dtype=float
        )
        self._tmp_Eww_norm_unref = numpy.array(
            numpy.zeros([self.bins[0], self.bins[1], self.bins[2]]), dtype=float
        )
        self._tmp_Dipole_x_dens = numpy.array(
            numpy.zeros([self.bins[0], self.bins[1], self.bins[2]]), dtype=float
        )
        self._tmp_Dipole_y_dens = numpy.array(
            numpy.zeros([self.bins[0], self.bins[1], self.bins[2]]), dtype=float
        )
        self._tmp_Dipole_z_dens = numpy.array(
            numpy.zeros([self.bins[0], self.bins[1], self.bins[2]]), dtype=float
        )
        self._tmp_Dipole_dens = numpy.array(
            numpy.zeros([self.bins[0], self.bins[1], self.bins[2]]), dtype=float
        )
        self._tmp_Neighbor_dens = numpy.array(
            numpy.zeros([self.bins[0], self.bins[1], self.bins[2]]), dtype=float
        )
        self._tmp_Neighbor_norm = numpy.array(
            numpy.zeros([self.bins[0], self.bins[1], self.bins[2]]), dtype=float
        )
        self._tmp_Order_norm = numpy.array(
            numpy.zeros([self.bins[0], self.bins[1], self.bins[2]]), dtype=float
        )

        for x_index, x in enumerate(
            range(int(cut_bins[0, 0]), int(cut_bins[0, 0]) + self.bins[0])
        ):
            for y_index, y in enumerate(
                range(int(cut_bins[1, 0]), int(cut_bins[1, 0]) + self.bins[1])
            ):
                for z_index, z in enumerate(
                    range(int(cut_bins[2, 0]), int(cut_bins[2, 0]) + self.bins[2])
                ):
                    self._tmp_Pop[x_index][y_index][z_index] = self.Pop[x, y, z]
                    self._tmp_gO[x_index][y_index][z_index] = self.gO[x, y, z]
                    self._tmp_gH[x_index][y_index][z_index] = self.gH[x, y, z]
                    self._tmp_dTStrans_dens[x_index][y_index][z_index] = (
                        self.dTStrans_dens[x, y, z]
                    )
                    self._tmp_dTStrans_norm[x_index][y_index][z_index] = (
                        self.dTStrans_norm[x, y, z]
                    )
                    self._tmp_dTSorient_dens[x_index][y_index][z_index] = (
                        self.dTSorient_dens[x, y, z]
                    )
                    self._tmp_dTSorient_norm[x_index][y_index][z_index] = (
                        self.dTSorient_norm[x, y, z]
                    )
                    if self.gist17:
                        self._tmp_dTSsix_dens[x_index][y_index][z_index] = (
                            self.dTSsix_dens[x, y, z]
                        )
                        self._tmp_dTSsix_norm[x_index][y_index][z_index] = (
                            self.dTSsix_norm[x, y, z]
                        )
                    self._tmp_Esw_dens[x_index][y_index][z_index] = self.Esw_dens[
                        x, y, z
                    ]
                    self._tmp_Esw_norm[x_index][y_index][z_index] = self.Esw_norm[
                        x, y, z
                    ]
                    self._tmp_Eww_dens[x_index][y_index][z_index] = self.Eww_dens[
                        x, y, z
                    ]
                    self._tmp_Eww_norm_unref[x_index][y_index][z_index] = (
                        self.Eww_norm_unref[x, y, z]
                    )
                    self._tmp_Dipole_x_dens[x_index][y_index][z_index] = (
                        self.Dipole_x_dens[x, y, z]
                    )
                    self._tmp_Dipole_y_dens[x_index][y_index][z_index] = (
                        self.Dipole_y_dens[x, y, z]
                    )
                    self._tmp_Dipole_z_dens[x_index][y_index][z_index] = (
                        self.Dipole_z_dens[x, y, z]
                    )
                    self._tmp_Dipole_dens[x_index][y_index][z_index] = self.Dipole_dens[
                        x, y, z
                    ]
                    self._tmp_Neighbor_dens[x_index][y_index][z_index] = (
                        self.Neighbor_dens[x, y, z]
                    )
                    self._tmp_Neighbor_norm[x_index][y_index][z_index] = (
                        self.Neighbor_norm[x, y, z]
                    )
                    self._tmp_Order_norm[x_index][y_index][z_index] = self.Order_norm[
                        x, y, z
                    ]

        self._update()

    def get_nan(self) -> numpy.ndarray:
        temporary = numpy.zeros(self.bins, dtype=bool)
        temporary[numpy.isnan(self.Pop)] = True
        return temporary

    def get_pop(self) -> numpy.ndarray:
        temporary = numpy.zeros(self.bins, dtype=bool)
        temporary[numpy.where(self.Pop > 0)] = True
        return temporary

    def write_maps(self, prefix: str = "gist", pymol: bool = True):
        data_dict = collections.OrderedDict()

        data_dict["_Pop.dx"] = [self.Pop, 1.0]
        data_dict["_gO.dx"] = [self.gO, 4.0]
        data_dict["_gH.dx"] = [self.gH, 4.0]
        data_dict["_dTStrans_dens.dx"] = [self.dTStrans_dens, 0.2]
        data_dict["_dTStrans_norm.dx"] = [self.dTStrans_norm, 1.0]
        data_dict["_dTSorient_dens.dx"] = [self.dTSorient_dens, 0.2]
        data_dict["_dTSorient_norm.dx"] = [self.dTSorient_norm, 1.0]
        if self.gist17:
            data_dict["_dTSsix_dens.dx"] = [self.dTSsix_dens, 0.2]
            data_dict["_dTSsix_norm.dx"] = [self.dTSsix_norm, 1.0]
        data_dict["_Esw_dens.dx"] = [self.Esw_dens, 0.2]
        data_dict["_Esw_norm.dx"] = [self.Esw_norm, 1.0]
        data_dict["_Eww_dens.dx"] = [self.Eww_dens, 0.2]
        data_dict["_Eww_norm_unref.dx"] = [self.Eww_norm_unref, 1.0]
        data_dict["_Eww_norm_ref.dx"] = [self.ref_ene - self.Eww_norm_unref, 1.0]
        data_dict["_Eww_norm_ref_dens.dx"] = [
            (self.ref_ene - self.Eww_norm_unref) * self.gO * self.ref_rho,
            1.0,
        ]
        data_dict["_Dipole_x_dens.dx"] = [self.Dipole_x_dens, 1.0]
        data_dict["_Dipole_y_dens.dx"] = [self.Dipole_y_dens, 1.0]
        data_dict["_Dipole_z_dens.dx"] = [self.Dipole_z_dens, 1.0]
        data_dict["_Dipole_dens.dx"] = [self.Dipole_dens, 1.0]
        data_dict["_Neighbor_dens.dx"] = [self.Neighbor_dens, 0.5]
        data_dict["_Neighbor_norm.dx"] = [self.Neighbor_norm, 1.0]
        data_dict["_Order_norm.dx"] = [self.Order_norm, 5.5]
        data_dict["_dTS_dens.dx"] = [
            self.dTStrans_dens + self.dTSorient_dens,
            0.2,
        ]
        data_dict["_dTS_norm.dx"] = [
            self.dTStrans_norm + self.dTSorient_norm,
            1.0,
        ]
        data_dict["_E_dens.dx"] = [self.Esw_dens + self.Eww_dens, 0.2]
        data_dict["_E_norm.dx"] = [self.Esw_norm + self.Eww_norm_unref, 1.0]
        data_dict["_Neighbor_loss_norm.dx"] = [
            self.Neighbor_norm - self.bulk_NN,
            0.5,
        ]

        if pymol:
            pymol_string = ""
            pymol_string += "from pymol import cmd\n"
            pymol_string += "from collections import OrderedDict\n"
            pymol_string += "\n"

        for name, data in data_dict.items():
            write_files(
                Frac2Real=self.get_nice_frac2real(),
                Bins=self.bins,
                Origin=self.origin,
                Value=data[0],
                Format="DX",
                Filename=prefix + name,
                Nan_fill=-999,
            )

            if pymol:
                new_name = str(prefix + name).replace(".dx", "")
                pymol_string += "### %s ###\n" % new_name
                pymol_string += 'cmd.load("./%s")\n' % (prefix + name)
                pymol_string += 'cmd.isomesh("%s", "%s", level=%s)\n' % (
                    new_name + "_map",
                    new_name,
                    data[1],
                )
                pymol_string += 'cmd.map_double("%s")\n' % (new_name)
                pymol_string += "\n"

        if pymol:
            pymol_string += 'cmd.disable("*_map")\n'
            pymol_string += 'cmd.do("color blue, *_map")\n'
            pymol_string += 'cmd.do("set mesh_negative_color, red")\n'
            pymol_string += 'cmd.do("set mesh_negative_visible")\n'

            with open(prefix + "_pymol.py", "w") as file_handle:
                file_handle.write(pymol_string)


class loadgist(gist):
    def __init__(self, Path: str, gist17: bool = False):
        gist.__init__(
            self,
            Bins=numpy.array([50, 50, 50]),
            Origin=numpy.array([0, 0, 0]),
            Delta=numpy.array([0.5, 0.5, 0.5]),
            gist17=gist17,
        )

        if not os.path.exists(Path):
            raise IOError("File %s not found." % Path)

        self.path = Path

        if Path.endswith(".gz"):
            map_file_ref = gzip.open(Path, "r")
        else:
            map_file_ref = open(Path, "r")
        map_file = map_file_ref.readlines()
        map_file_ref.close()

        start_row = -1

        for line_index, item in enumerate(map_file):
            if (
                len(item.rstrip().split()) > 1
                and item.rstrip().split()[0] == "voxel"
                and item.rstrip().split()[1] == "xcoord"
            ):
                start_row = line_index + 1
                break

        z_start = map_file[start_row].rstrip().split()[3]
        y_start = map_file[start_row].rstrip().split()[2]
        x_start = map_file[start_row].rstrip().split()[1]

        found_bins_z = False
        found_bins_y = False
        found_bins_x = False

        for line_index, line in enumerate(map_file[start_row + 1 :]):
            if (
                found_bins_z
                and not found_bins_y
                and line.rstrip().split()[2] == y_start
            ):
                self.bins[1] = (line_index + 1) // self.bins[2]
                break

            if not found_bins_z and line.rstrip().split()[3] == z_start:
                self.bins[2] = line_index + 1
                found_bins_z = True

        self.bins[0] = (len(map_file) - start_row) // (self.bins[2] * self.bins[1])
        self.n = numpy.copy(self.bins)

        self.delta = numpy.array(
            [
                float(
                    map_file[start_row + 1 + self.bins[0] * self.bins[1]]
                    .rstrip()
                    .split()[1]
                )
                - float(map_file[start_row].rstrip().split()[1]),
                float(map_file[start_row + 1 + self.bins[0]].rstrip().split()[2])
                - float(map_file[start_row].rstrip().split()[2]),
                float(map_file[start_row + 1].rstrip().split()[3])
                - float(map_file[start_row].rstrip().split()[3]),
            ]
        )

        self.dim = self.bins * self.delta

        self.origin = -self.delta * 0.5 + numpy.array(
            [
                float(map_file[start_row].rstrip().split()[1]),
                float(map_file[start_row].rstrip().split()[2]),
                float(map_file[start_row].rstrip().split()[3]),
            ]
        )

        self.frac2real = numpy.array(
            [
                [self.dim[0] / self.n[0], 0.0, 0.0],
                [0.0, self.dim[1] / self.n[1], 0.0],
                [0.0, 0.0, self.dim[2] / self.n[2]],
            ]
        )

        self.real2frac = numpy.linalg.inv(self.frac2real)
        self.rotation_matrix = numpy.eye(3, 3)
        self.translation_vector = numpy.zeros(3)
        self.center = self.get_real(self.bins / 2)

        line_index = 0

        if self.gist17:
            skip_col = 2
        else:
            skip_col = 0

        for x in range(0, self.bins[0]):
            for y in range(0, self.bins[1]):
                for z in range(0, self.bins[2]):
                    self._tmp_Pop[x][y][z] = float(
                        map_file[start_row + line_index].rstrip().split()[4]
                    )
                    self._tmp_gO[x][y][z] = float(
                        map_file[start_row + line_index].rstrip().split()[5]
                    )
                    self._tmp_gH[x][y][z] = float(
                        map_file[start_row + line_index].rstrip().split()[6]
                    )
                    self._tmp_dTStrans_dens[x][y][z] = float(
                        map_file[start_row + line_index].rstrip().split()[7]
                    )
                    self._tmp_dTStrans_norm[x][y][z] = float(
                        map_file[start_row + line_index].rstrip().split()[8]
                    )
                    self._tmp_dTSorient_dens[x][y][z] = float(
                        map_file[start_row + line_index].rstrip().split()[9]
                    )
                    self._tmp_dTSorient_norm[x][y][z] = float(
                        map_file[start_row + line_index].rstrip().split()[10]
                    )
                    if self.gist17:
                        self._tmp_dTSsix_dens[x][y][z] = float(
                            map_file[start_row + line_index].rstrip().split()[11]
                        )
                        self._tmp_dTSsix_norm[x][y][z] = float(
                            map_file[start_row + line_index].rstrip().split()[12]
                        )
                    self._tmp_Esw_dens[x][y][z] = float(
                        map_file[start_row + line_index].rstrip().split()[11 + skip_col]
                    )
                    self._tmp_Esw_norm[x][y][z] = float(
                        map_file[start_row + line_index].rstrip().split()[12 + skip_col]
                    )
                    self._tmp_Eww_dens[x][y][z] = float(
                        map_file[start_row + line_index].rstrip().split()[13 + skip_col]
                    )
                    self._tmp_Eww_norm_unref[x][y][z] = float(
                        map_file[start_row + line_index].rstrip().split()[14 + skip_col]
                    )
                    self._tmp_Dipole_x_dens[x][y][z] = float(
                        map_file[start_row + line_index].rstrip().split()[15 + skip_col]
                    )
                    self._tmp_Dipole_y_dens[x][y][z] = float(
                        map_file[start_row + line_index].rstrip().split()[16 + skip_col]
                    )
                    self._tmp_Dipole_z_dens[x][y][z] = float(
                        map_file[start_row + line_index].rstrip().split()[17 + skip_col]
                    )
                    self._tmp_Dipole_dens[x][y][z] = float(
                        map_file[start_row + line_index].rstrip().split()[18 + skip_col]
                    )
                    self._tmp_Neighbor_dens[x][y][z] = float(
                        map_file[start_row + line_index].rstrip().split()[19 + skip_col]
                    )
                    self._tmp_Neighbor_norm[x][y][z] = float(
                        map_file[start_row + line_index].rstrip().split()[20 + skip_col]
                    )
                    self._tmp_Order_norm[x][y][z] = float(
                        map_file[start_row + line_index].rstrip().split()[21 + skip_col]
                    )

                    line_index += 1

        self._update()


class write_files(object):
    def __init__(
        self,
        Delta: typing.Optional[numpy.ndarray] = None,
        Frac2Real: typing.Optional[numpy.ndarray] = None,
        Bins: typing.Optional[numpy.ndarray] = None,
        Origin: typing.Optional[numpy.ndarray] = None,
        Value: typing.Optional[numpy.ndarray] = None,
        XYZ: typing.Optional[numpy.ndarray] = None,
        X: typing.Optional[numpy.ndarray] = None,
        Y: typing.Optional[numpy.ndarray] = None,
        Z: typing.Optional[numpy.ndarray] = None,
        Format: str = "PDB",
        Filename: typing.Optional[str] = None,
        Nan_fill: float = -1.0,
    ):
        self._delta = Delta
        self._frac2real = Frac2Real
        self._bins = Bins
        self._origin = Origin
        self._value = Value
        self._x = X
        self._y = Y
        self._z = Z
        self._format = Format
        self._filename = Filename
        self._xyz = XYZ
        self._nan_fill = Nan_fill

        if type(self._filename) != str:
            self._filename = "output."
            self._filename += self._format

        self._writers = {
            "PDB": self._write_PDB,
            "DX": self._write_DX,
            "GIST": self._write_GIST,
        }

        data = self._writers[self._format]()

        output_file = open(self._filename, "w")
        output_file.write(data)
        output_file.close()

    def _merge_x_y_z(self) -> numpy.ndarray:
        return numpy.stack((self._x, self._y, self._z), axis=1)

    def _write_PDB(self) -> str:
        if io_helpers.are_you_numpy(self._xyz):
            if self._xyz.shape[-1] != 3:
                raise TypeError("XYZ array has wrong shape.")
        else:
            if not (
                io_helpers.are_you_numpy(self._x)
                or io_helpers.are_you_numpy(self._y)
                or io_helpers.are_you_numpy(self._z)
            ):
                raise TypeError(
                    "If XYZ is not given, x,y and z coordinates must be given in separate arrays."
                )
            else:
                self._xyz = self._merge_x_y_z()

        if self._value is None:
            self._value = numpy.zeros(len(self._xyz), dtype=float)

        data = "REMARK File written by write_files.py\n"

        for xyz_index, xyz in enumerate(self._xyz):
            chain_id = string.ascii_uppercase[(len(str(xyz_index + 1)) // 5)]
            atom_counts = xyz_index - (len(str(xyz_index + 1)) // 6) * 100000
            resi_counts = xyz_index - (len(str(xyz_index + 1)) // 5) * 10000
            data += (
                "%-6s%5d %4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          \n"
                % (
                    "HETATM",
                    atom_counts + 1,
                    "X",
                    "",
                    "MAP",
                    chain_id,
                    resi_counts + 1,
                    "",
                    xyz[0],
                    xyz[1],
                    xyz[2],
                    0.00,
                    float(self._value[xyz_index]),
                )
            )

        data += "END\n"
        return data

    def _write_DX(self) -> str:
        if not (
            io_helpers.are_you_numpy(self._origin)
            or io_helpers.are_you_numpy(self._bins)
        ):
            raise TypeError("Origin and bins must be given.")

        if io_helpers.are_you_numpy(self._delta) == io_helpers.are_you_numpy(
            self._frac2real
        ):
            raise TypeError("Either delta or frac2real must be given.")

        if io_helpers.are_you_numpy(self._delta):
            self._frac2real = numpy.zeros((3, 3), dtype=float)
            numpy.fill_diagonal(self._frac2real, self._delta)

        data = """object 1 class gridpositions counts %d %d %d
origin %8.4f %8.4f %8.4f
delta %8.4f %8.4f %8.4f
delta %8.4f %8.4f %8.4f
delta %8.4f %8.4f %8.4f
object 2 class gridconnections counts %d %d %d
object 3 class array type float rank 0 items %d data follows
""" % (
            self._bins[0],
            self._bins[1],
            self._bins[2],
            self._origin[0],
            self._origin[1],
            self._origin[2],
            self._frac2real[0][0],
            self._frac2real[0][1],
            self._frac2real[0][2],
            self._frac2real[1][0],
            self._frac2real[1][1],
            self._frac2real[1][2],
            self._frac2real[2][0],
            self._frac2real[2][1],
            self._frac2real[2][2],
            self._bins[0],
            self._bins[1],
            self._bins[2],
            self._bins[2] * self._bins[1] * self._bins[0],
        )

        item_index = 0
        for x_index in range(0, self._bins[0]):
            for y_index in range(0, self._bins[1]):
                for z_index in range(0, self._bins[2]):
                    if numpy.isnan(self._value[x_index][y_index][z_index]):
                        data += str(self._nan_fill) + " "
                    else:
                        if self._value[x_index][y_index][z_index] == 0.0:
                            data += "0 "
                        else:
                            data += str(self._value[x_index][y_index][z_index]) + " "

                    item_index += 1
                    if item_index == 3:
                        data += "\n"
                        item_index = 0
        return data

    def _write_GIST(self) -> typing.Optional[str]:
        pass


class PDB(object):
    def __init__(self, Path: str):
        self.path = Path
        self.crd: list[list[float]] = []
        self.B: list[str] = []

        with open(self.path, "r") as pdb_file:
            for line_index, line in enumerate(pdb_file):
                if not (line[0:6].rstrip() == "ATOM" or line[0:6].rstrip() == "HETATM"):
                    continue

                if line_index <= 9999:
                    self.crd.append([])
                    self.crd[-1].append(float(line.rstrip()[30:38]))
                    self.crd[-1].append(float(line.rstrip()[38:46]))
                    self.crd[-1].append(float(line.rstrip()[46:54]))
                    self.B.append(line.rstrip()[54:59])

                if 9999 < line_index <= 99999:
                    self.crd.append([])
                    self.crd[-1].append(float(line.rstrip()[31:39]))
                    self.crd[-1].append(float(line.rstrip()[39:47]))
                    self.crd[-1].append(float(line.rstrip()[47:55]))
                    self.B.append(line.rstrip()[55:60])

                if line_index > 99999:
                    self.crd.append([])
                    self.crd[-1].append(float(line.rstrip()[33:41]))
                    self.crd[-1].append(float(line.rstrip()[41:49]))
                    self.crd[-1].append(float(line.rstrip()[49:57]))
                    self.B.append(line.rstrip()[57:62])

        self.crd_array = numpy.array(self.crd)
        self.B_array = numpy.array(self.B)


## methods


def guess_field(
    coordinates: numpy.ndarray,
    delta: numpy.ndarray = numpy.array([0.5, 0.5, 0.5]),
) -> field:
    center = numpy.mean(coordinates, axis=0)
    minimum = numpy.min((center - coordinates), axis=0)
    maximum = numpy.max((center - coordinates), axis=0)
    bins = numpy.rint(numpy.abs(maximum - minimum) / delta + (5.0 / delta))
    del minimum, maximum
    return field(Bins=bins, Delta=delta, Center=center)
