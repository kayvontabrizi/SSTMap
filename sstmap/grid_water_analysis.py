##############################################################################
#  SSTMap: A Python library for the calculation of water structure and
#         thermodynamics on solute surfaces from molecular dynamics
#         trajectories.
# MIT License
# Copyright 2016-2017 Lehman College City University of New York and the Authors
#
# Authors: Kamran Haider, Steven Ramsay, Anthony Cruz Balberdy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################

## imports

# standard
import sys
import typing

# custom
import mdtraj
import numpy

# local
import _sstmap_ext
from sstmap import utils
from sstmap import water_analysis


## constants

GASKCAL = 0.0019872041


## classes


class GridWaterAnalysis(water_analysis.WaterAnalysis):
    @utils.function_timer
    def __init__(
        self,
        topology_file: str,
        trajectory: str,
        start_frame: int = 0,
        num_frames: int = 0,
        supporting_file: typing.Optional[str] = None,
        ligand_file: typing.Optional[str] = None,
        grid_center: typing.Optional[list[float]] = None,
        grid_dimensions: list[int] = [20, 20, 20],
        grid_resolution: list[float] = [0.5, 0.5, 0.5],
        rho_bulk: float = 0.0334,
        prefix: str = "test",
    ):
        print("Initializing ...")
        self.start_frame = start_frame
        self.num_frames = num_frames
        self.rho_bulk = float(rho_bulk)
        super(GridWaterAnalysis, self).__init__(
            topology_file, trajectory, supporting_file
        )

        self.grid_dims = numpy.asarray(grid_dimensions, int)
        self.resolution = grid_resolution[0]
        self.prefix = prefix
        if ligand_file is None and grid_center is None:
            sys.exit(
                "Please provide value of the grid center as a list of x, y, z coordinates or "
                "specify a ligand PDB file whose center would be chosen as grid center."
            )

        if ligand_file is not None and grid_center is None:
            ligand = mdtraj.load_pdb(ligand_file, no_boxchk=True)
            center_of_mass = numpy.zeros((ligand.n_frames, 3))
            masses = numpy.ones(ligand.n_atoms)
            masses /= masses.sum()
            center_of_mass[0, :] = ligand.xyz[0, :].astype("float64").T.dot(masses)
            grid_center = center_of_mass[0, :] * 10.0
        self.voxel_vol = self.resolution**3.0

        self.initialize_grid(grid_center, grid_resolution, grid_dimensions)

        self.voxeldata, self.voxel_quarts, self.voxel_O_coords = (
            self.initialize_voxel_data()
        )

    def initialize_grid(
        self,
        center: typing.Optional[list[float]],
        resolution: list[float],
        dimensions: list[int],
    ):
        print("Initializing ...")
        self.center = numpy.array(center, dtype=float)
        self.dims = numpy.array(dimensions, dtype=int)
        self.spacing = numpy.array(resolution, dtype=float)
        self.gridmax = self.dims * self.spacing + 1.5
        origin = self.center - (0.5 * self.dims * self.spacing)
        self.origin = numpy.around(origin, decimals=3)

        length = numpy.array(self.dims / self.spacing, dtype=float)
        self.grid_size = numpy.ceil((length / self.spacing) + 1.0)
        self.grid_size = self.grid_size.astype(numpy.uint32)

        self.grid = numpy.zeros(self.dims, dtype=int)

    def initialize_voxel_data(
        self,
    ) -> tuple[numpy.ndarray, list[list[float]], list[list[float]]]:
        voxel_count = 0
        voxel_array = numpy.zeros((self.grid.size, 35), dtype="float64")
        for index, value in numpy.ndenumerate(self.grid):
            _index = numpy.array(index, dtype=numpy.int32)
            point = _index * self.spacing + self.origin + 0.5 * self.spacing
            voxel_array[voxel_count, 1] = point[0]
            voxel_array[voxel_count, 2] = point[1]
            voxel_array[voxel_count, 3] = point[2]
            voxel_array[voxel_count, 0] = voxel_count
            voxel_count += 1
        voxel_quarts: list[list[float]] = [[] for _ in range(voxel_array.shape[0])]
        voxel_O_coords: list[list[float]] = [[] for _ in range(voxel_array.shape[0])]
        return voxel_array, voxel_quarts, voxel_O_coords

    def calculate_euler_angles(self, water: tuple[int, int], coords: numpy.ndarray):
        xlab = numpy.asarray([1.0, 0.0, 0.0])
        zlab = numpy.asarray([0.0, 0.0, 1.0])

        voxel_id = water[0]
        oxygen_water = coords[water[1], :]

        hydrogen1_water = coords[water[1] + 1, :] - oxygen_water
        hydrogen2_water = coords[water[1] + 2, :] - oxygen_water

        hydrogen1_water /= numpy.linalg.norm(hydrogen1_water)
        hydrogen2_water /= numpy.linalg.norm(hydrogen2_water)
        axis_rotation_1 = numpy.cross(hydrogen1_water, xlab)
        signed_axis_rotation = numpy.copy(axis_rotation_1)
        axis_rotation_1 /= numpy.linalg.norm(axis_rotation_1)
        dot_product_1 = numpy.sum(xlab * hydrogen1_water)
        theta = numpy.arccos(dot_product_1)
        sign = numpy.sum(signed_axis_rotation * hydrogen1_water)
        if sign > 0:
            theta /= 2.0
        else:
            theta /= -2.0

        w1 = numpy.cos(theta)
        sin_theta = numpy.sin(theta)
        x1 = axis_rotation_1[0] * sin_theta
        y1 = axis_rotation_1[1] * sin_theta
        z1 = axis_rotation_1[2] * sin_theta
        w2 = w1
        x2 = x1
        y2 = y1
        z2 = z1

        hydrogen_temp = numpy.zeros(3)
        hydrogen_temp[0] = (
            (w2 * w2 + x2 * x2) - (y2 * y2 + z2 * z2)
        ) * hydrogen1_water[0]
        hydrogen_temp[0] = (
            2 * (x2 * y2 + w2 * z2) * hydrogen1_water[1]
        ) + hydrogen_temp[0]
        hydrogen_temp[0] = (
            2 * (x2 * z2 - w2 * y2) * hydrogen1_water[2]
        ) + hydrogen_temp[0]

        hydrogen_temp[1] = 2 * (x2 * y2 - w2 * z2) * hydrogen1_water[0]
        hydrogen_temp[1] = (
            (w2 * w2 - x2 * x2 + y2 * y2 - z2 * z2) * hydrogen1_water[1]
        ) + hydrogen_temp[1]
        hydrogen_temp[1] = (
            2 * (y2 * z2 + w2 * x2) * hydrogen1_water[2]
        ) + hydrogen_temp[1]

        hydrogen_temp[2] = 2 * (x2 * z2 + w2 * y2) * hydrogen1_water[0]
        hydrogen_temp[2] = (
            2 * (y2 * z2 - w2 * x2) * hydrogen1_water[1]
        ) + hydrogen_temp[2]
        hydrogen_temp[2] = (
            (w2 * w2 - x2 * x2 - y2 * y2 + z2 * z2) * hydrogen1_water[2]
        ) + hydrogen_temp[2]

        hydrogen_temp2 = numpy.zeros(3)
        hydrogen_temp2[0] = (
            (w2 * w2 + x2 * x2) - (y2 * y2 + z2 * z2)
        ) * hydrogen2_water[0]
        hydrogen_temp2[0] = (
            2 * (x2 * y2 + w2 * z2) * hydrogen2_water[1]
        ) + hydrogen_temp2[0]
        hydrogen_temp2[0] = (
            2 * (x2 * z2 - w2 * y2) * hydrogen2_water[2]
        ) + hydrogen_temp2[0]

        hydrogen_temp2[1] = 2 * (x2 * y2 - w2 * z2) * hydrogen2_water[0]
        hydrogen_temp2[1] = (
            (w2 * w2 - x2 * x2 + y2 * y2 - z2 * z2) * hydrogen2_water[1]
        ) + hydrogen_temp2[1]
        hydrogen_temp2[1] = (
            2 * (y2 * z2 + w2 * x2) * hydrogen2_water[2]
        ) + hydrogen_temp2[1]

        hydrogen_temp2[2] = 2 * (x2 * z2 + w2 * y2) * hydrogen2_water[0]
        hydrogen_temp2[2] = (
            2 * (y2 * z2 - w2 * x2) * hydrogen2_water[1]
        ) + hydrogen_temp2[2]
        hydrogen_temp2[2] = (
            (w2 * w2 - x2 * x2 - y2 * y2 + z2 * z2) * hydrogen2_water[2]
        ) + hydrogen_temp2[2]

        axis_rotation_2 = numpy.cross(hydrogen_temp, hydrogen_temp2)
        axis_rotation_2 /= numpy.linalg.norm(axis_rotation_2)
        dot_product_2 = numpy.sum(axis_rotation_2 * zlab)
        theta = numpy.arccos(dot_product_2)

        signed_axis_rotation = numpy.cross(axis_rotation_2, zlab)
        sign = numpy.sum(signed_axis_rotation * hydrogen_temp)

        if sign < 0:
            theta /= 2.0
        else:
            theta /= -2.0

        w3 = numpy.cos(theta)
        sin_theta = numpy.sin(theta)
        x3 = xlab[0] * sin_theta
        y3 = xlab[1] * sin_theta
        z3 = xlab[2] * sin_theta

        w4 = w1 * w3 - x1 * x3 - y1 * y3 - z1 * z3
        x4 = w1 * x3 + x1 * w3 + y1 * z3 - z1 * y3
        y4 = w1 * y3 - x1 * z3 + y1 * w3 + z1 * x3
        z4 = w1 * z3 + x1 * y3 - y1 * x3 + z1 * w3
        self.voxel_quarts[voxel_id].extend([w4, x4, y4, z4])
        self.voxel_O_coords[voxel_id].extend(oxygen_water)

    @utils.function_timer
    def calculate_entropy(self, num_frames: typing.Optional[int] = None):
        if num_frames is None:
            num_frames = self.num_frames
        _sstmap_ext.getNNTrEntropy(
            num_frames,
            self.voxel_vol,
            self.rho_bulk,
            300.0,
            self.grid_dims,
            self.voxeldata,
            self.voxel_O_coords,
            self.voxel_quarts,
        )

    def _process_frame(
        self,
        trajectory: mdtraj.Trajectory,
        energy: bool,
        hbonds: bool,
        entropy: bool,
    ):
        neighbor_cutoff_squared = 3.5**2
        trajectory.xyz *= 10.0
        coords = trajectory.xyz
        unit_cell = trajectory.unitcell_vectors[0] * 10.0
        waters: list[tuple[int, int]] = []
        _sstmap_ext.assign_voxels(
            trajectory.xyz,
            self.dims,
            self.gridmax,
            self.origin,
            waters,
            self.wat_oxygen_atom_ids,
        )

        distance_matrix = numpy.zeros((self.water_sites, self.all_atom_ids.shape[0]))

        for water in waters:
            self.voxeldata[water[0], 4] += 1
            if energy or hbonds:
                energy_lj_array, energy_elec_array = numpy.copy(
                    self.acoeff
                ), numpy.copy(self.chg_product)

                valid_neighbors = numpy.ones(self.neighbor_ids.shape[0], dtype=bool)
                valid_neighbors[numpy.where(self.neighbor_ids == water)] = False
                neighbor_ids = self.neighbor_ids[valid_neighbors]
                water_neighbors_shell = self.wat_nbrs_shell[valid_neighbors]
                _sstmap_ext.get_pairwise_distances(
                    water,
                    self.all_atom_ids,
                    numpy.array([neighbor_cutoff_squared]),
                    neighbor_ids,
                    water_neighbors_shell,
                    coords,
                    unit_cell,
                    distance_matrix,
                    0,
                )
                water_neighbors = self.all_atom_ids[
                    numpy.where(water_neighbors_shell == 0)
                ]
                self.voxeldata[water[0], 19] += water_neighbors.shape[0]
                _sstmap_ext.calculate_energy(
                    water[1],
                    distance_matrix,
                    energy_elec_array,
                    energy_lj_array,
                    self.bcoeff,
                )

                if self.prot_atom_ids.shape[0] != 0:
                    self.voxeldata[water[0], 13] += numpy.sum(
                        energy_lj_array[:, self.non_water_atom_ids]
                    )
                    self.voxeldata[water[0], 13] += numpy.sum(
                        energy_elec_array[:, self.non_water_atom_ids]
                    )

                self.voxeldata[water[0], 15] += numpy.sum(
                    energy_lj_array[:, self.wat_oxygen_atom_ids[0] : water[1]]
                ) + numpy.sum(energy_lj_array[:, water[1] + self.water_sites :])
                self.voxeldata[water[0], 15] += numpy.sum(
                    energy_elec_array[:, self.wat_oxygen_atom_ids[0] : water[1]]
                ) + numpy.sum(energy_elec_array[:, water[1] + self.water_sites :])
                energy_neighbor_list = [
                    numpy.sum(
                        energy_lj_array[:, water_neighbors + index]
                        + energy_elec_array[:, water_neighbors + index]
                    )
                    for index in range(self.water_sites)
                ]
                self.voxeldata[water[0], 17] += numpy.sum(energy_neighbor_list)

                self.wat_nbrs_shell[valid_neighbors] = water_neighbors_shell
                self.neighbor_ids[valid_neighbors] = neighbor_ids

                if hbonds:
                    protein_neighbors_all = self.prot_atom_ids[
                        numpy.where(
                            distance_matrix[0, :][self.prot_atom_ids]
                            <= neighbor_cutoff_squared
                        )
                    ]
                    protein_neighbors_hb = protein_neighbors_all[
                        numpy.where(self.prot_hb_types[protein_neighbors_all] != 0)
                    ]

                    if water_neighbors.shape[0] > 0:
                        hb_water_water = self.calculate_hydrogen_bonds(
                            trajectory, water[1], water_neighbors
                        )
                        acceptor_water_water = hb_water_water[:, 0][
                            numpy.where(hb_water_water[:, 0] == water[1])
                        ].shape[0]
                        donor_water_water = (
                            hb_water_water.shape[0] - acceptor_water_water
                        )
                        self.voxeldata[water[0], 25] += hb_water_water.shape[0]
                        self.voxeldata[water[0], 31] += donor_water_water
                        self.voxeldata[water[0], 33] += acceptor_water_water
                        if (
                            water_neighbors.shape[0] != 0
                            and hb_water_water.shape[0] != 0
                        ):
                            self.voxeldata[water[0], 21] += (
                                water_neighbors.shape[0] / hb_water_water.shape[0]
                            )

                    if protein_neighbors_hb.shape[0] > 0:
                        hb_solute_water = self.calculate_hydrogen_bonds(
                            trajectory,
                            water[1],
                            protein_neighbors_hb,
                            water_water=False,
                        )
                        acceptor_solute_water = hb_solute_water[:, 0][
                            numpy.where(hb_solute_water[:, 0] == water[1])
                        ].shape[0]
                        donor_solute_water = (
                            hb_solute_water.shape[0] - acceptor_solute_water
                        )
                        self.voxeldata[water[0], 23] += hb_solute_water.shape[0]
                        self.voxeldata[water[0], 27] += donor_solute_water
                        self.voxeldata[water[0], 29] += acceptor_solute_water

            if entropy:
                self.calculate_euler_angles(water, coords[0, :, :])

    @utils.function_timer
    def calculate_grid_quantities(
        self,
        energy: bool = True,
        entropy: bool = True,
        hbonds: bool = True,
    ):
        utils.print_progress_bar(0, self.num_frames)
        if not self.topology_file.endswith(".h5"):
            topology = mdtraj.load_topology(self.topology_file)
        read_num_frames = 0
        with mdtraj.open(self.trajectory) as trajectory_file:
            for frame_index in range(
                self.start_frame, self.start_frame + self.num_frames
            ):
                utils.print_progress_bar(
                    frame_index - self.start_frame, self.num_frames
                )
                trajectory_file.seek(frame_index)
                if not self.trajectory.endswith(".h5"):
                    trajectory = trajectory_file.read_as_traj(
                        topology, n_frames=1, stride=1
                    )
                else:
                    trajectory = trajectory_file.read_as_traj(n_frames=1, stride=1)
                if trajectory.n_frames == 0:
                    print("No more frames to read.")
                    break
                else:
                    self._process_frame(trajectory, energy, hbonds, entropy)
                    read_num_frames += 1
            if read_num_frames < self.num_frames:
                print(
                    (
                        "{0:d} frames found in the trajectory, resetting self.num_frames.".format(
                            read_num_frames
                        )
                    )
                )
                self.num_frames = read_num_frames

        for voxel in range(self.voxeldata.shape[0]):
            if self.voxeldata[voxel, 4] > 1.0:
                self.voxeldata[voxel, 14] = self.voxeldata[voxel, 13] / (
                    self.voxeldata[voxel, 4] * 2.0
                )
                self.voxeldata[voxel, 13] /= self.num_frames * self.voxel_vol * 2.0
                self.voxeldata[voxel, 16] = self.voxeldata[voxel, 15] / (
                    self.voxeldata[voxel, 4] * 2.0
                )
                self.voxeldata[voxel, 15] /= self.num_frames * self.voxel_vol * 2.0
                if self.voxeldata[voxel, 19] > 0.0:
                    self.voxeldata[voxel, 18] = self.voxeldata[voxel, 17] / (
                        self.voxeldata[voxel, 19] * 2.0
                    )
                    self.voxeldata[voxel, 17] /= (
                        self.num_frames
                        * self.voxel_vol
                        * self.voxeldata[voxel, 19]
                        * 2.0
                    )
                for index in range(19, 35, 2):
                    self.voxeldata[voxel, index + 1] = (
                        self.voxeldata[voxel, index] / self.voxeldata[voxel, 4]
                    )
                    self.voxeldata[voxel, index] /= self.num_frames * self.voxel_vol
            else:
                self.voxeldata[voxel, 13] *= 0.0
                self.voxeldata[voxel, 15] *= 0.0
                if self.voxeldata[voxel, 19] > 0.0:
                    self.voxeldata[voxel, 17] *= 0.0
                for index in range(19, 35, 2):
                    self.voxeldata[voxel, index] *= 0.0

        if entropy:
            self.calculate_entropy(num_frames=self.num_frames)

    @utils.function_timer
    def write_data(self, prefix: typing.Optional[str] = None):
        if prefix is None:
            prefix = self.prefix
        print("Writing voxel data ...")
        with open(prefix + "_gist_data.txt", "w") as output_file:
            gist_header = "voxel x y z nwat gO dTStr-dens dTStr-norm dTSor-dens dTSor-norm dTSsix-dens dTSsix-norm Esw-dens Esw-norm Eww-dens Eww-norm Eww-nbr-dens Eww-nbr-norm Nnbr-dens Nnbr-norm fHB-dens fHB-norm Nhbsw_dens Nhbsw_norm Nhbww_dens Nhbww_norm Ndonsw_dens Ndonsw_norm Naccsw_dens Naccsw_norm Ndonww_dens Ndonww_norm Naccww_dens Naccww_norm\n"
            output_file.write(gist_header)
            formatted_output_occupied_voxels = (
                "{0[0]:.0f} {0[1]:.3f} {0[2]:.3f} {0[3]:.3f} {0[4]:.0f} {0[5]:.6f} "
            )
            formatted_output_one_voxels = formatted_output_occupied_voxels
            formatted_output_empty_voxels = (
                "{0[0]:.0f} {0[1]:.3f} {0[2]:.3f} {0[3]:.3f} {0[4]:.0f} {0[5]:.0f} "
            )
            for quantity_index in range(7, 35):
                formatted_output_occupied_voxels += "{0[%d]:.6f} " % quantity_index
                formatted_output_empty_voxels += "{0[%d]:.0f} " % quantity_index
                if quantity_index in [19, 20]:
                    formatted_output_one_voxels += "{0[%d]:.3f} " % quantity_index
                elif quantity_index in [7, 8, 11, 12]:
                    formatted_output_one_voxels += "{0[%d]:.6f} " % quantity_index
                else:
                    formatted_output_one_voxels += "{0[%d]:.0f} " % quantity_index
            formatted_output_occupied_voxels += "\n"
            formatted_output_one_voxels += "\n"
            formatted_output_empty_voxels += "\n"
            for voxel_index in range(self.voxeldata.shape[0]):
                if self.voxeldata[voxel_index, 4] == 0.0:
                    output_file.write(
                        formatted_output_empty_voxels.format(
                            self.voxeldata[voxel_index, :]
                        )
                    )
                elif self.voxeldata[voxel_index, 4] == 1.0:
                    mask_one_voxel_data = numpy.zeros(
                        self.voxeldata[voxel_index, :].shape[0]
                    )
                    mask_one_voxel_data[
                        list(range(0, 6)) + [7, 8, 11, 12] + list(range(19, 21))
                    ] = self.voxeldata[
                        voxel_index,
                        list(range(0, 6)) + [7, 8, 11, 12] + list(range(19, 21)),
                    ]
                    output_file.write(
                        formatted_output_one_voxels.format(mask_one_voxel_data)
                    )
                else:
                    output_file.write(
                        formatted_output_occupied_voxels.format(
                            self.voxeldata[voxel_index, :]
                        )
                    )

    @utils.function_timer
    def generate_dx_files(self, prefix: typing.Optional[str] = None):
        if prefix is None:
            prefix = self.prefix
        print("Generating dx files ...")
        gist_header = "voxel x y z nwat gO gH dTStrans-dens dTStrans-norm dTSorient-dens dTSorient-norm dTSsix-dens dTSsix-norm Esw-dens Esw-norm Eww-dens Eww-norm Eww-nbr-dens Eww-nbr-norm neighbor-dens neighbor-norm fHB-dens fHB-norm Nhbww-dens Nhbww-norm Nhbsw-dens Nhbsw-norm Ndonsw-dens Ndonsw-norm Naccsw-dens Naccsw-norm Ndonww-dens Ndonww-norm Naccww-dens Naccww-norm\n"
        dx_header = ""
        dx_header += "object 1 class gridpositions counts %d %d %d\n" % (
            self.grid.shape[0],
            self.grid.shape[1],
            self.grid.shape[2],
        )
        dx_header += "origin %.3f %.3f %.3f\n" % (
            self.origin[0],
            self.origin[1],
            self.origin[2],
        )
        dx_header += "delta %.1f 0 0\n" % (self.spacing[0])
        dx_header += "delta 0 %.1f 0\n" % (self.spacing[1])
        dx_header += "delta 0 0 %.1f\n" % (self.spacing[2])
        dx_header += "object 2 class gridconnections counts %d %d %d\n" % (
            self.grid.shape[0],
            self.grid.shape[1],
            self.grid.shape[2],
        )
        dx_header += (
            "object 3 class array type double rank 0 items %d data follows\n"
            % (self.grid.shape[0] * self.grid.shape[1] * self.grid.shape[2])
        )
        dx_file_objects: list[typing.Optional[typing.TextIO]] = []

        data_keys = gist_header.strip("\n").split()

        for data_field, title in enumerate(data_keys):
            if data_field > 4 and data_field % 2 == 1 and title != "gH":
                file_object = open(prefix + "_" + title + ".dx", "w")
                file_object.write(dx_header)
                dx_file_objects.append(file_object)
            else:
                dx_file_objects.append(None)

        for voxel_index in range(1, len(self.voxeldata) + 1):
            for column_index in range(5, len(data_keys), 2):
                dx_file_objects[column_index].write(
                    "%g " % (self.voxeldata[voxel_index - 1][column_index])
                )
                if voxel_index % 3 == 0:
                    dx_file_objects[column_index].write("\n")

        for file_object in dx_file_objects:
            if file_object is not None:
                file_object.close()

    def print_system_summary(self):
        print("System information:")
        print(("\tParameter file: %s\n" % self.topology_file))
        print(("\tTrajectory: %s\n" % self.trajectory))
        print(
            (
                "\tFrames: %d, Total Atoms: %d, Waters: %d, Solute Atoms: %d\n"
                % (
                    self.num_frames,
                    self.all_atom_ids.shape[0],
                    self.wat_oxygen_atom_ids.shape[0],
                    self.non_water_atom_ids.shape[0],
                )
            )
        )
        print("Grid information:")
        print(
            (
                "\tGIST grid center: %5.3f %5.3f %5.3f\n"
                % (self.center[0], self.center[1], self.center[2])
            )
        )
        print(
            (
                "\tGIST grid dimensions: %i %i %i\n"
                % (self.dims[0], self.dims[1], self.dims[2])
            )
        )
        print(("\tGIST grid spacing: %5.3f A^3\n" % (self.spacing[0])))

    def print_calcs_summary(self, num_frames: typing.Optional[int] = None):
        if num_frames is None:
            num_frames = self.num_frames
        print("Summary of main calculations:")
        number_of_waters_grid = 0.0
        energy_solute_water_total = 0.0
        energy_water_water_total = 0.0
        for voxel in self.voxeldata:
            if voxel[4] > 1.0:
                number_of_waters_grid += voxel[4] / (num_frames * self.voxel_vol)
                energy_solute_water_total += voxel[13]
                energy_water_water_total += voxel[15]

        number_of_waters_grid *= self.voxel_vol
        energy_solute_water_total *= self.voxel_vol * 2.0
        energy_water_water_total *= self.voxel_vol
        print(("Number of frames processed: %d" % num_frames))
        print(
            (
                "\tAverage number of water molecules over the grid: %d"
                % number_of_waters_grid
            )
        )
        print(
            (
                "\tTotal Solute-Water Energy over the grid: %.6f"
                % energy_solute_water_total
            )
        )
        print(
            (
                "\tTotal Water-Water Energy over the grid: %.6f"
                % energy_water_water_total
            )
        )
