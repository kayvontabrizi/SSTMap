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
import os
import shutil
import sys
import typing

# custom
import mdtraj
import numpy
import scipy.spatial

# local
import _sstmap_entropy
import _sstmap_ext
import _sstmap_probableconfig
from sstmap import utils
from sstmap import water_analysis


## classes


class SiteWaterAnalysis(water_analysis.WaterAnalysis):
    @utils.function_timer
    def __init__(
        self,
        topology_file: str,
        trajectory: str,
        start_frame: int = 0,
        num_frames: typing.Optional[int] = None,
        supporting_file: typing.Optional[str] = None,
        ligand_file: typing.Optional[str] = None,
        hsa_region_radius: float = 5.0,
        clustercenter_file: typing.Optional[str] = None,
        rho_bulk: float = 0.0334,
        prefix: str = "hsa",
    ):
        print("Initializing ...")
        self.start_frame = start_frame
        self.num_frames = num_frames
        super(SiteWaterAnalysis, self).__init__(
            topology_file, trajectory, supporting_file
        )

        self.prefix = prefix
        self.site_waters: typing.Optional[list[list[tuple[int, int]]]] = None
        if clustercenter_file is None and ligand_file is None:
            sys.exit(
                "Please provide either a ligand file for clustering or "
                "a cluster center file to generate hydration sites."
            )
        if self.num_frames == 0:
            sys.exit(
                "Number of frames = %d, no calculations will be performed"
                % self.num_frames
            )

        self.rho_bulk = float(rho_bulk)
        self.ligand = ligand_file
        self.clustercenter_file = clustercenter_file
        self.hsa_region_radius = hsa_region_radius * 0.1
        if hsa_region_radius > 10.0:
            print(
                "Warning: Currently, clustering region is restricted to a 10.0A sphere around the ligand molecule."
            )
            self.hsa_region_radius = 10.0
        self.hsa_data: typing.Optional[numpy.ndarray] = None
        self.hsa_dict: typing.Optional[dict[int, list[typing.Any]]] = None
        self.is_site_waters_populated = False
        self.hsa_region_O_ids: list[list[int]] = []
        self.hsa_region_flat_ids: list[list[int]] = []
        self.hsa_region_water_coords: typing.Optional[numpy.ndarray] = None
        self.data_titles = [
            "index",
            "x",
            "y",
            "z",
            "nwat",
            "occupancy",
            "Esw",
            "EswLJ",
            "EswElec",
            "Eww",
            "EwwLJ",
            "EwwElec",
            "Etot",
            "Ewwnbr",
            "TSsw_trans",
            "TSsw_orient",
            "TStot",
            "Nnbrs",
            "Nhbww",
            "Nhbsw",
            "Nhbtot",
            "f_hb_ww",
            "f_enc",
            "Acc_ww",
            "Don_ww",
            "Acc_sw",
            "Don_sw",
            "solute_acceptors",
            "solute_donors",
        ]
        self.energy_ww_lr_breakdown: typing.Optional[list[list[float]]] = None
        self.angular_st_distribution: typing.Optional[
            list[list[tuple[numpy.ndarray, numpy.ndarray]]]
        ] = None

    @utils.function_timer
    def initialize_hydration_sites(self, clustering_density_cutoff: float = 2.0):
        cluster_coords, self.site_waters = self.generate_clusters(
            clustering_density_cutoff, self.ligand, self.clustercenter_file
        )
        self.hsa_data, self.hsa_dict = self.initialize_site_data(cluster_coords)
        self.is_site_waters_populated = True

    @utils.function_timer
    def initialize_site_data(
        self, cluster_coords: numpy.ndarray
    ) -> tuple[numpy.ndarray, dict[int, list[typing.Any]]]:
        number_of_sites = cluster_coords.shape[0]
        site_array = numpy.zeros((number_of_sites, len(self.data_titles)))
        site_dict: dict[int, list[typing.Any]] = {}
        for site_index in range(number_of_sites):
            site_array[site_index, 0] = site_index
            site_array[site_index, 1] = cluster_coords[site_index, 0]
            site_array[site_index, 2] = cluster_coords[site_index, 1]
            site_array[site_index, 3] = cluster_coords[site_index, 2]
            site_dict[site_index] = [[] for _ in range(len(self.data_titles))]
            site_dict[site_index].append(numpy.zeros((self.num_frames * 3, 3)))
        return site_array, site_dict

    @utils.function_timer
    def generate_clusters(
        self,
        density_factor: float,
        ligand_file: typing.Optional[str],
        clustercenter_file: typing.Optional[str],
    ) -> tuple[numpy.ndarray, list[list[tuple[int, int]]]]:
        sphere_radius = mdtraj.utils.in_units_of(1.0, "angstroms", "nanometers")
        if not self.topology_file.endswith(".h5"):
            topology = mdtraj.load_topology(self.topology_file)
        if self.non_water_atom_ids.shape[0] == 0:
            raise Exception(
                ValueError,
                "Clustering is supported only for solute-solvent systems, no solute atoms found.",
            )

        ligand = mdtraj.load_pdb(ligand_file, no_boxchk=True)
        ligand_coords = ligand.xyz[0, :, :]
        binding_site_atom_indices = numpy.asarray(list(range(ligand_coords.shape[0])))
        init_cluster_coords = None

        if clustercenter_file is None:
            clustering_stride = 10
            print("Reading trajectory for clustering.")
            with mdtraj.open(self.trajectory) as trajectory_file:
                trajectory_file.seek(self.start_frame)
                if self.num_frames is None:
                    if not self.trajectory.endswith(".h5"):
                        trajectory_short = trajectory_file.read_as_traj(
                            topology,
                            atom_indices=numpy.concatenate(
                                (binding_site_atom_indices, self.wat_oxygen_atom_ids)
                            ),
                            stride=clustering_stride,
                        )
                    else:
                        trajectory_short = trajectory_file.read_as_traj(
                            atom_indices=numpy.concatenate(
                                (binding_site_atom_indices, self.wat_oxygen_atom_ids)
                            ),
                            stride=clustering_stride,
                        )
                else:
                    if not self.trajectory.endswith(".h5"):
                        trajectory_short = trajectory_file.read_as_traj(
                            topology,
                            atom_indices=numpy.concatenate(
                                (binding_site_atom_indices, self.wat_oxygen_atom_ids)
                            ),
                            n_frames=self.num_frames,
                            stride=clustering_stride,
                        )
                    else:
                        trajectory_short = trajectory_file.read_as_traj(
                            atom_indices=numpy.concatenate(
                                (binding_site_atom_indices, self.wat_oxygen_atom_ids)
                            ),
                            n_frames=self.num_frames,
                            stride=clustering_stride,
                        )
                if trajectory_short.n_frames < 10:
                    sys.exit(
                        "Clustering requires at least 100 frames, current trajectory contains {0:d} frames.".format(
                            trajectory_short.n_frames
                        )
                    )
                print(
                    "Performing an initial clustering over {0:d} frames.".format(
                        trajectory_short.n_frames
                    )
                )

                coords = trajectory_short.xyz
                for frame_index in range(trajectory_short.n_frames):
                    for pseudo_index in range(binding_site_atom_indices.shape[0]):
                        coords[frame_index, pseudo_index, :] = ligand_coords[
                            pseudo_index, :
                        ]

                haystack = numpy.setdiff1d(
                    trajectory_short.topology.select("all"), binding_site_atom_indices
                )
                binding_site_waters = mdtraj.compute_neighbors(
                    trajectory_short,
                    self.hsa_region_radius,
                    binding_site_atom_indices,
                    haystack_indices=haystack,
                )

                water_id_frame_list = [
                    (frame_index, neighbor)
                    for frame_index in range(len(binding_site_waters))
                    for neighbor in binding_site_waters[frame_index]
                ]

                water_coordinates = numpy.ma.array(
                    [coords[water[0], water[1], :] for water in water_id_frame_list],
                    mask=False,
                )
                tree = scipy.spatial.cKDTree(water_coordinates)
                neighbor_list = tree.query_ball_point(water_coordinates, sphere_radius)
                neighbor_count_list = numpy.ma.array(
                    [len(neighbors) for neighbors in neighbor_list], mask=False
                )
                cutoff = trajectory_short.n_frames * density_factor * 0.1401
                if numpy.ceil(cutoff) - cutoff <= 0.5:
                    cutoff = numpy.ceil(cutoff)
                else:
                    cutoff = numpy.floor(cutoff)
                number_of_waters = 3 * cutoff

                cluster_list: list[tuple[int, int]] = []
                cluster_iteration = 0
                while number_of_waters > cutoff:
                    max_index = numpy.argmax(neighbor_count_list)
                    to_exclude = numpy.array(neighbor_list[max_index])
                    number_of_waters = len(to_exclude) + 1

                    neighbor_count_list.mask[to_exclude] = True
                    neighbor_count_list.mask[max_index] = True
                    water_coordinates.mask[to_exclude] = True
                    water_coordinates.mask[max_index] = True

                    neighbors_of_excluded = numpy.unique(
                        numpy.array(
                            [
                                excluded_neighbor
                                for excluded_neighbors in neighbor_list[to_exclude]
                                for excluded_neighbor in excluded_neighbors
                            ]
                        )
                    )

                    to_update = numpy.setxor1d(to_exclude, neighbors_of_excluded)
                    to_update = numpy.setdiff1d(to_update, numpy.asarray(max_index))

                    if to_update.shape[0] != 0:
                        tree = scipy.spatial.cKDTree(water_coordinates)
                        updated_neighbor_list = tree.query_ball_point(
                            water_coordinates[to_update], sphere_radius
                        )
                        for index, neighbors in enumerate(updated_neighbor_list):
                            if not neighbor_count_list.mask[to_update[index]]:
                                neighbor_count_list[to_update[index]] = len(neighbors)

                    current_water = water_id_frame_list[max_index]
                    current_water_coords = mdtraj.utils.in_units_of(
                        coords[current_water[0], current_water[1], :],
                        "nanometers",
                        "angstroms",
                    )
                    near_flag = 0
                    if len(cluster_list) != 0:
                        for cluster in cluster_list:
                            cluster_coords = coords[cluster[0], cluster[1], :]
                            distance = numpy.linalg.norm(
                                current_water_coords - cluster_coords
                            )
                            if distance < 1.20:
                                near_flag += 1
                    if near_flag == 0:
                        cluster_iteration += 1
                        cluster_list.append(water_id_frame_list[max_index])
                init_cluster_coords = [
                    coords[cluster[0], cluster[1], :] for cluster in cluster_list
                ]
        else:
            clusters_pdb_file = mdtraj.load_pdb(clustercenter_file, no_boxchk=True)
            init_cluster_coords = clusters_pdb_file.xyz[0, :, :]

        print("Reading trajectory to obtain water molecules for each cluster.")
        with mdtraj.open(self.trajectory) as trajectory_file:
            trajectory_file.seek(self.start_frame)
            if self.num_frames is None:
                if not self.trajectory.endswith(".h5"):
                    trajectory = trajectory_file.read_as_traj(
                        topology,
                        stride=1,
                        atom_indices=numpy.concatenate(
                            (binding_site_atom_indices, self.wat_oxygen_atom_ids)
                        ),
                    )
                    self.num_frames = trajectory.n_frames
                else:
                    trajectory = trajectory_file.read_as_traj(
                        stride=1,
                        atom_indices=numpy.concatenate(
                            (binding_site_atom_indices, self.wat_oxygen_atom_ids)
                        ),
                    )
                    self.num_frames = trajectory.n_frames
            else:
                if not self.trajectory.endswith(".h5"):
                    trajectory = trajectory_file.read_as_traj(
                        topology,
                        n_frames=self.num_frames,
                        stride=1,
                        atom_indices=numpy.concatenate(
                            (binding_site_atom_indices, self.wat_oxygen_atom_ids)
                        ),
                    )
                else:
                    trajectory = trajectory_file.read_as_traj(
                        n_frames=self.num_frames,
                        stride=1,
                        atom_indices=numpy.concatenate(
                            (binding_site_atom_indices, self.wat_oxygen_atom_ids)
                        ),
                    )
                if trajectory.n_frames < self.num_frames:
                    print(
                        (
                            "Warning: {0:d} frames found in the trajectory, resetting self.num_frames.".format(
                                trajectory.n_frames
                            )
                        )
                    )
                    self.num_frames = trajectory.n_frames
            for frame_index in range(trajectory.n_frames):
                for pseudo_index in range(binding_site_atom_indices.shape[0]):
                    trajectory.xyz[frame_index, pseudo_index, :] = ligand_coords[
                        pseudo_index, :
                    ]
            haystack = numpy.setdiff1d(
                trajectory.topology.select("all"), binding_site_atom_indices
            )
            start_point = haystack[0]
            binding_site_waters = mdtraj.compute_neighbors(
                trajectory,
                self.hsa_region_radius,
                binding_site_atom_indices,
                haystack_indices=haystack,
            )

            start = 0
            for frame_index in range(len(binding_site_waters)):
                self.hsa_region_O_ids.append([])
                self.hsa_region_flat_ids.append([])
                for water in binding_site_waters[frame_index]:
                    water_offset_index = water - start_point
                    water_offset = (
                        water_offset_index * self.water_sites
                    ) + self.wat_oxygen_atom_ids[0]
                    self.hsa_region_O_ids[frame_index].append(water_offset)
                    self.hsa_region_flat_ids[frame_index].append(start)
                    start += 3

            water_id_frame_list = [
                (frame_index, neighbor)
                for frame_index in range(len(binding_site_waters))
                for neighbor in binding_site_waters[frame_index]
            ]
            water_coordinates = numpy.array(
                [trajectory.xyz[water[0], water[1], :] for water in water_id_frame_list]
            )

        self.hsa_region_water_coords = numpy.zeros(
            (len(water_id_frame_list) * 3, 3), dtype=float
        )
        tree = scipy.spatial.cKDTree(water_coordinates)
        neighbor_list = tree.query_ball_point(init_cluster_coords, sphere_radius)
        final_cluster_coords: list[numpy.ndarray] = []
        cutoff = int(self.num_frames * density_factor * 0.1401)
        if numpy.ceil(cutoff) - cutoff <= 0.5:
            cutoff = numpy.ceil(cutoff)
        else:
            cutoff = numpy.floor(cutoff)

        if clustercenter_file is None:
            print(
                (
                    "Refining initial cluster positions by considering {0:d} frames.".format(
                        self.num_frames
                    )
                )
            )

            site_waters: list[list[tuple[int, int]]] = []
            cluster_index = 1
            for cluster in neighbor_list:
                cluster_water_coords = water_coordinates[cluster]
                if len(cluster) > cutoff:
                    near_flag = 0
                    waters_offset = [
                        (
                            water_id_frame_list[water][0] + self.start_frame,
                            (
                                (water_id_frame_list[water][1] - start_point)
                                * self.water_sites
                            )
                            + self.wat_oxygen_atom_ids[0],
                        )
                        for water in cluster
                    ]

                    center_of_mass = numpy.zeros(3)
                    masses = numpy.ones(cluster_water_coords.shape[0])
                    masses /= masses.sum()
                    center_of_mass[:] = water_coordinates[cluster].T.dot(masses)
                    cluster_center = center_of_mass[:]

                    for other, coord in enumerate(final_cluster_coords[:-1]):
                        distance = numpy.linalg.norm(
                            mdtraj.utils.in_units_of(
                                cluster_center, "nanometers", "angstroms"
                            )
                            - coord
                        )
                        if distance < 1.20:
                            near_flag += 1

                    if near_flag == 0:
                        final_cluster_coords.append(
                            mdtraj.utils.in_units_of(
                                cluster_center, "nanometers", "angstroms"
                            )
                        )
                        site_waters.append(waters_offset)
                        cluster_index += 1
        else:
            final_cluster_coords = mdtraj.utils.in_units_of(
                init_cluster_coords, "nanometers", "angstroms"
            )
            site_waters = []
            cluster_index = 1
            for cluster in neighbor_list:
                waters_offset = [
                    (
                        water_id_frame_list[water][0] + self.start_frame,
                        (
                            (water_id_frame_list[water][1] - start_point)
                            * self.water_sites
                        )
                        + self.wat_oxygen_atom_ids[0],
                    )
                    for water in cluster
                ]
                site_waters.append(waters_offset)
                cluster_index += 1

        utils.write_watpdb_from_coords("clustercenterfile", final_cluster_coords)
        self.clustercenter_file = "clustercenterfile.pdb"
        print(("Final number of clusters: {0:d}".format(len(final_cluster_coords))))
        return numpy.asarray(final_cluster_coords), site_waters

    def _process_frame(
        self,
        trajectory: mdtraj.Trajectory,
        frame_index: int,
        energy: bool,
        hbonds: bool,
        entropy: bool,
        energy_lr_breakdown: bool,
        angular_structure: bool,
        shell_radii: typing.Optional[list[float]],
        r_theta_cutoff: float,
    ):
        site_waters_copy = list(self.site_waters)
        neighbor_cutoff_squared = 3.5**2
        neighbor_cutoff_index = -1
        neighbor_cutoff_found = False
        shell_radii_copy: list[float] = []

        if shell_radii is None:
            shell_radii_copy.append(neighbor_cutoff_squared)
            neighbor_cutoff_index = 0
        else:
            for shell in shell_radii:
                shell_radii_copy.append(shell)
            shell_radii_copy.sort()

            for radius_index, radius in enumerate(shell_radii_copy):
                if (
                    radius > neighbor_cutoff_squared - 0.0001
                    and radius < neighbor_cutoff_squared + 0.0001
                ):
                    neighbor_cutoff_index = radius_index
                    neighbor_cutoff_found = True
            if neighbor_cutoff_index == -1:
                shell_radii_copy.append(neighbor_cutoff_squared)
                shell_radii_copy.sort()
                for radius_index, radius in enumerate(shell_radii_copy):
                    if (
                        radius > neighbor_cutoff_squared - 0.0001
                        and radius < neighbor_cutoff_squared + 0.0001
                    ):
                        neighbor_cutoff_index = radius_index

        trajectory.xyz *= 10.0
        coords = trajectory.xyz
        trajectory.unitcell_lengths *= 10.0
        unit_cell = trajectory.unitcell_vectors[0] * 10.0

        distance_matrix = numpy.zeros((self.water_sites, self.all_atom_ids.shape[0]))
        for site_index in range(self.hsa_data.shape[0]):
            water_oxygen = None
            if self.is_site_waters_populated:
                if len(site_waters_copy[site_index]) != 0:
                    if site_waters_copy[site_index][0][0] == frame_index:
                        water_oxygen = site_waters_copy[site_index].pop(0)[1]
                        index = int(self.hsa_data[site_index, 4]) * 3
                        index_pairs = list(
                            zip(
                                list(range(water_oxygen, water_oxygen + 3)),
                                list(range(index, index + 3)),
                            )
                        )
                        for index_pair in index_pairs:
                            self.hsa_dict[site_index][-1][index_pair[1]] += coords[
                                0, index_pair[0], :
                            ]
                        self.hsa_data[site_index, 4] += 1

            if water_oxygen is not None and (energy or hbonds):
                valid_neighbors = numpy.ones(self.neighbor_ids.shape[0], dtype=bool)
                valid_neighbors[numpy.where(self.neighbor_ids == water_oxygen)] = False
                neighbor_ids = self.neighbor_ids[valid_neighbors]
                water_neighbors_shell = self.wat_nbrs_shell[valid_neighbors]
                _sstmap_ext.get_pairwise_distances(
                    numpy.asarray([site_index, water_oxygen]),
                    self.all_atom_ids,
                    numpy.array(shell_radii_copy),
                    neighbor_ids,
                    water_neighbors_shell,
                    coords,
                    unit_cell,
                    distance_matrix,
                    0,
                )
                water_neighbors = neighbor_ids[
                    numpy.where(water_neighbors_shell < (neighbor_cutoff_index + 1))
                ]
                self.hsa_dict[site_index][17].append(water_neighbors.shape[0])
                if energy:
                    energy_lj_array, energy_elec_array = numpy.copy(
                        self.acoeff
                    ), numpy.copy(self.chg_product)
                    _sstmap_ext.calculate_energy(
                        water_oxygen,
                        distance_matrix,
                        energy_elec_array,
                        energy_lj_array,
                        self.bcoeff,
                    )

                    energy_lj_solute_water = numpy.sum(
                        energy_lj_array[:, self.non_water_atom_ids]
                    )
                    energy_elec_solute_water = numpy.sum(
                        energy_elec_array[:, self.non_water_atom_ids]
                    )

                    energy_lj_water_water_left = energy_lj_array[
                        :, self.wat_oxygen_atom_ids[0] : water_oxygen
                    ]
                    energy_lj_water_water_right = energy_lj_array[
                        :, water_oxygen + self.water_sites :
                    ]
                    energy_lj_water_water = numpy.sum(
                        energy_lj_water_water_left
                    ) + numpy.sum(energy_lj_water_water_right)
                    energy_elec_water_water_left = energy_elec_array[
                        :, self.wat_oxygen_atom_ids[0] : water_oxygen
                    ]
                    energy_elec_water_water_right = energy_elec_array[
                        :, water_oxygen + self.water_sites :
                    ]
                    energy_elec_water_water = numpy.sum(
                        energy_elec_water_water_left
                    ) + numpy.sum(energy_elec_water_water_right)
                    energy_neighbor_list = [
                        numpy.sum(
                            energy_lj_array[:, neighbor : neighbor + self.water_sites]
                            + energy_elec_array[
                                :, neighbor : neighbor + self.water_sites
                            ]
                        )
                        for neighbor in water_neighbors
                    ]

                    energy_lj_water_water = numpy.sum(
                        energy_lj_array[:, self.wat_oxygen_atom_ids[0] : water_oxygen]
                    ) + numpy.sum(energy_lj_array[:, water_oxygen + self.water_sites :])
                    energy_elec_water_water = numpy.sum(
                        energy_elec_array[:, self.wat_oxygen_atom_ids[0] : water_oxygen]
                    ) + numpy.sum(
                        energy_elec_array[:, water_oxygen + self.water_sites :]
                    )

                    self.hsa_dict[site_index][7].append(energy_lj_solute_water)
                    self.hsa_dict[site_index][8].append(energy_elec_solute_water)
                    self.hsa_dict[site_index][10].append(energy_lj_water_water)
                    self.hsa_dict[site_index][11].append(energy_elec_water_water)
                    self.hsa_dict[site_index][6].append(
                        energy_lj_solute_water + energy_elec_solute_water
                    )
                    self.hsa_dict[site_index][9].append(
                        energy_lj_water_water + energy_elec_water_water
                    )
                    self.hsa_dict[site_index][12].append(
                        energy_lj_solute_water
                        + energy_elec_solute_water
                        + energy_lj_water_water
                        + energy_elec_water_water
                    )
                    self.hsa_dict[site_index][13].extend(energy_neighbor_list)

                    if energy_lr_breakdown:
                        if neighbor_cutoff_found:
                            self.energy_ww_lr_breakdown[site_index][
                                neighbor_cutoff_index
                            ] += sum(energy_neighbor_list)
                        shift = 0
                        for radius_index, radius in enumerate(shell_radii_copy):
                            if radius_index == neighbor_cutoff_index:
                                continue
                            water_neighbors_shell_index = neighbor_ids[
                                numpy.where(water_neighbors_shell == radius_index)
                            ]
                            if (
                                not neighbor_cutoff_found
                                and radius_index == neighbor_cutoff_index + 1
                            ):
                                water_neighbors_shell_index = numpy.concatenate(
                                    (water_neighbors_shell_index, water_neighbors)
                                )
                                shift = 1
                            if water_neighbors_shell_index.shape[0] > 0:
                                energy_neighbor_list = [
                                    numpy.sum(
                                        energy_lj_array[
                                            :, neighbor : neighbor + self.water_sites
                                        ]
                                        + energy_elec_array[
                                            :, neighbor : neighbor + self.water_sites
                                        ]
                                    )
                                    for neighbor in water_neighbors_shell_index
                                ]
                                self.energy_ww_lr_breakdown[site_index][
                                    radius_index - shift
                                ] += sum(energy_neighbor_list)

                    self.wat_nbrs_shell[valid_neighbors] = water_neighbors_shell
                    self.neighbor_ids[valid_neighbors] = neighbor_ids

                if hbonds:
                    hbond_total = 0
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
                            trajectory, water_oxygen, water_neighbors
                        )
                        acceptor_water_water = hb_water_water[:, 0][
                            numpy.where(hb_water_water[:, 0] == water_oxygen)
                        ].shape[0]
                        donor_water_water = (
                            hb_water_water.shape[0] - acceptor_water_water
                        )
                        self.hsa_dict[site_index][18].append(hb_water_water.shape[0])
                        self.hsa_dict[site_index][23].append(acceptor_water_water)
                        self.hsa_dict[site_index][24].append(donor_water_water)
                        hbond_total += hb_water_water.shape[0]
                        if (
                            water_neighbors.shape[0] != 0
                            and hb_water_water.shape[0] != 0
                        ):
                            self.hsa_dict[site_index][21].append(
                                hb_water_water.shape[0] / water_neighbors.shape[0]
                            )
                    if protein_neighbors_hb.shape[0] > 0:
                        hb_solute_water = self.calculate_hydrogen_bonds(
                            trajectory,
                            water_oxygen,
                            protein_neighbors_hb,
                            water_water=False,
                        )
                        acceptor_solute_water = hb_solute_water[:, 0][
                            numpy.where(hb_solute_water[:, 0] == water_oxygen)
                        ].shape[0]
                        donor_solute_water = (
                            hb_solute_water.shape[0] - acceptor_solute_water
                        )
                        donor_solute_water_ids = hb_solute_water[:, 1][
                            numpy.where(hb_solute_water[:, 0] == water_oxygen)
                        ]
                        acceptor_solute_water_ids = hb_solute_water[:, 0][
                            numpy.where(hb_solute_water[:, 0] != water_oxygen)
                        ]
                        self.hsa_dict[site_index][19].append(hb_solute_water.shape[0])
                        self.hsa_dict[site_index][25].append(acceptor_solute_water)
                        self.hsa_dict[site_index][26].append(donor_solute_water)
                        self.hsa_dict[site_index][27].extend(acceptor_solute_water_ids)
                        self.hsa_dict[site_index][28].extend(donor_solute_water_ids)
                        hbond_total += hb_solute_water.shape[0]
                    self.hsa_dict[site_index][20].append(hbond_total)
                    if angular_structure:
                        water_neighbors = self.wat_oxygen_atom_ids[
                            numpy.where(
                                (
                                    distance_matrix[0, :][self.wat_oxygen_atom_ids]
                                    <= r_theta_cutoff**2
                                )
                                & (
                                    distance_matrix[0, :][self.wat_oxygen_atom_ids]
                                    > 0.0
                                )
                            )
                        ]
                        angles = self.water_nbr_orientations(
                            trajectory, water_oxygen, water_neighbors
                        )
                        distances = numpy.sqrt(distance_matrix[0, water_neighbors])
                        self.angular_st_distribution[site_index].extend(
                            zip(distances, angles)
                        )

        if entropy:
            for index, water_oxygen in enumerate(
                self.hsa_region_O_ids[frame_index - self.start_frame]
            ):
                flat_id = self.hsa_region_flat_ids[frame_index - self.start_frame][
                    index
                ]
                index_pairs = list(
                    zip(
                        list(range(water_oxygen, water_oxygen + 3)),
                        list(range(flat_id, flat_id + 3)),
                    )
                )
                for index_pair in index_pairs:
                    self.hsa_region_water_coords[index_pair[1], :] += coords[
                        0, index_pair[0], :
                    ]

    @utils.function_timer
    def calculate_site_quantities(
        self,
        energy: bool = True,
        entropy: bool = True,
        hbonds: bool = True,
        energy_lr_breakdown: bool = False,
        angular_structure: bool = False,
        shell_radii: typing.Optional[list[float]] = None,
        r_theta_cutoff: float = 6.0,
    ):
        utils.print_progress_bar(0, self.num_frames)
        if not self.trajectory.endswith(".h5"):
            topology = mdtraj.load_topology(self.topology_file)
        read_num_frames = 0
        if energy_lr_breakdown:
            if shell_radii is None:
                shell_radii = [3.5, 5.5, 8.5]
            shell_radii = [radius**2 for radius in shell_radii]
            self.energy_ww_lr_breakdown = [
                [0.0 for _ in shell_radii] for _ in range(self.hsa_data.shape[0])
            ]

        if angular_structure:
            if r_theta_cutoff > 8.0:
                print(
                    "Warning: r_theta_cutoff > 8.0 can take a long time. "
                    "Resetting angular structure distance cutoff to 8.0 Angstrom"
                )
                r_theta_cutoff = 8.0
            self.angular_st_distribution = [[] for _ in range(self.hsa_data.shape[0])]

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
                    self._process_frame(
                        trajectory,
                        frame_index,
                        energy,
                        hbonds,
                        entropy,
                        energy_lr_breakdown,
                        angular_structure,
                        shell_radii,
                        r_theta_cutoff,
                    )
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

        if entropy:
            self.generate_data_for_entropycalcs(self.start_frame, self.num_frames)
            self.run_entropy_scripts()
        self.normalize_site_quantities(self.num_frames)

    @utils.function_timer
    def generate_data_for_entropycalcs(
        self,
        start_frame: int,
        num_frames: int,
        user_defined_clusters: bool = False,
    ):
        print(
            "Writing PDB file containing all HSA region water molecules for entropy calculations."
        )
        utils.write_watpdb_from_coords(
            "within5Aofligand", self.hsa_region_water_coords, full_water_res=True
        )
        print("Done.")
        print("Writing PDB files for all water molecules in each hydration site.")
        for site_index in range(self.hsa_data.shape[0]):
            number_of_waters = int(self.hsa_data[site_index, 4]) * 3
            cluster_name = "{0:06d}".format(site_index + 1)
            utils.write_watpdb_from_coords(
                "cluster." + cluster_name,
                self.hsa_dict[site_index][-1][:number_of_waters, :],
                full_water_res=True,
            )
        print("Done.")

    @utils.function_timer
    def run_entropy_scripts(self, output_dir: typing.Optional[str] = None):
        current_directory = os.getcwd()
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            else:
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)

        else:
            output_dir = "entropy_output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            else:
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)

        input_cluster_argument = os.path.abspath(self.clustercenter_file)
        input_water_argument = os.path.abspath("within5Aofligand.pdb")

        os.chdir(current_directory + "/" + output_dir)
        print("Generating expanded cluster water files...")
        try:
            _sstmap_entropy.run_bruteclust(input_cluster_argument, input_water_argument)
        except Exception as exception:
            print(exception)
        os.chdir(current_directory)

        trans_dat, orient_dat = os.path.abspath("trans.dat"), os.path.abspath(
            "orient.dat"
        )
        if os.path.isfile(trans_dat):
            os.remove(trans_dat)
        if os.path.isfile(orient_dat):
            os.remove(orient_dat)

        input_output_argument = os.path.abspath(output_dir + "/probable.pdb")
        print("Running entropy calculation from extension module.")
        for site_index in range(self.hsa_data.shape[0]):
            cluster_filename = "cluster.{0:06d}.pdb".format(site_index + 1)
            input_index_argument = os.path.abspath(cluster_filename)
            input_expanded_argument = os.path.abspath(
                output_dir + "/" + cluster_filename
            )
            try:
                _sstmap_entropy.run_kdhsa102(
                    input_index_argument, input_expanded_argument
                )
                _sstmap_probableconfig.run_probconfig(
                    input_index_argument, input_output_argument
                )
            except Exception as exception:
                print(exception)

        coordinates = numpy.loadtxt(input_output_argument, usecols=(6, 7, 8))
        utils.write_watpdb_from_coords(
            "probable_configs", coordinates, full_water_res=True
        )

        if os.path.isfile(trans_dat) and os.path.isfile(orient_dat):
            trans_entropy, orient_entropy = numpy.loadtxt(trans_dat), numpy.loadtxt(
                orient_dat
            )
            if (
                trans_entropy.shape[0] == self.hsa_data.shape[0]
                and orient_entropy.shape[0] == self.hsa_data.shape[0]
            ):
                self.hsa_data[:, 14] += trans_entropy
                self.hsa_data[:, 15] += orient_entropy
                self.hsa_data[:, 16] += trans_entropy + orient_entropy
        shutil.rmtree(output_dir)
        os.remove(input_water_argument)

    @utils.function_timer
    def normalize_site_quantities(self, num_frames: int):
        sphere_volume = (4 / 3) * numpy.pi
        bulk_water_per_site = self.rho_bulk * sphere_volume * num_frames
        skip_normalization = [
            "index",
            "x",
            "y",
            "z",
            "nwat",
            "occupancy",
            "gO",
            "TSsw_trans",
            "TSsw_orient",
            "TStot",
            "solute_acceptors",
            "solute_donors",
        ]
        for site_index in range(self.hsa_data.shape[0]):
            number_of_waters = self.hsa_data[site_index, 4]
            if number_of_waters != 0:
                self.hsa_data[site_index, 5] = number_of_waters / (self.num_frames)
                for quantity_index in range(len(self.data_titles)):
                    if self.data_titles[quantity_index] not in skip_normalization:
                        if self.data_titles[quantity_index] in [
                            "Esw",
                            "EswLJ",
                            "EswElec",
                            "Eww",
                            "EwwLJ",
                            "EwwElec",
                            "Etot",
                        ]:
                            self.hsa_data[site_index, quantity_index] = (
                                numpy.sum(self.hsa_dict[site_index][quantity_index])
                                / number_of_waters
                            ) * 0.5
                        elif self.data_titles[quantity_index] in ["Ewwnbr"]:
                            if len(self.hsa_dict[site_index][17]) != 0:
                                self.hsa_data[site_index, quantity_index] = (
                                    numpy.sum(self.hsa_dict[site_index][quantity_index])
                                    / len(self.hsa_dict[site_index][quantity_index])
                                ) * 0.5
                        else:
                            self.hsa_data[site_index, quantity_index] = (
                                numpy.sum(self.hsa_dict[site_index][quantity_index])
                                / number_of_waters
                            )
                    if self.data_titles[quantity_index] in [
                        "solute_acceptors",
                        "solute_donors",
                    ]:
                        self.hsa_dict[site_index][quantity_index] = numpy.unique(
                            self.hsa_dict[site_index][quantity_index]
                        )
                    if self.data_titles[quantity_index] in ["f_enc"]:
                        self.hsa_data[site_index, quantity_index] = None
                if self.energy_ww_lr_breakdown is not None:
                    self.energy_ww_lr_breakdown[site_index] = [
                        (shell_energy / number_of_waters) * 0.5
                        for shell_energy in self.energy_ww_lr_breakdown[site_index]
                    ]

    def print_system_summary(self):
        print("System information:")
        print(("\tParameter file: %s\n" % self.topology_file))
        print(("\tTrajectory: %s\n" % self.trajectory))
        print(
            (
                "\tTotal Atoms: %d, Waters: %d, Solute Atoms: %d\n"
                % (
                    self.all_atom_ids.shape[0],
                    self.wat_oxygen_atom_ids.shape[0],
                    self.non_water_atom_ids.shape[0],
                )
            )
        )
        if self.hsa_data is not None:
            print(("\tNumber of clusters: %d\n" % len(self.hsa_data)))

    @utils.function_timer
    def write_calculation_summary(self):
        with open(self.prefix + "_hsa_summary.txt", "w") as output_file:
            header = " ".join(self.data_titles) + "\n"
            output_file.write(header)

            formatted_output = (
                "{0[0]:.0f} {0[1]:.2f} {0[2]:.2f} {0[3]:.2f} {0[4]:.0f} {0[5]:.2f} "
            )

            for quantity_index in range(6, len(self.data_titles) - 2):
                formatted_output += "{0[%d]:.6f} " % quantity_index

            formatted_output += "{1} {2}\n"
            for site_index in range(self.hsa_data.shape[0]):
                solute_acceptors = [
                    str(self.topology.atom(acceptor))
                    for acceptor in self.hsa_dict[site_index][27]
                ]
                solute_donors = [
                    str(self.topology.atom(donor))
                    for donor in self.hsa_dict[site_index][28]
                ]
                site_data_line = formatted_output.format(
                    self.hsa_data[site_index, :],
                    ",".join(solute_acceptors),
                    ",".join(solute_donors),
                )
                output_file.write(site_data_line)

    @utils.function_timer
    def write_data(self):
        skip_write_data = [
            "x",
            "y",
            "z",
            "nwat",
            "occupancy",
            "gO",
            "TSsw_trans",
            "TSsw_orient",
            "TStot",
            "f_enc",
            "solute_acceptors",
            "solute_donors",
        ]

        directory = self.prefix + "_hsa_data"
        if not os.path.exists(directory):
            os.makedirs(directory)

        for site_index in range(self.hsa_data.shape[0]):
            site_index_string = "/%03d_" % site_index
            for quantity_index in range(len(self.data_titles)):
                if (
                    self.data_titles[quantity_index] not in skip_write_data
                    and len(self.hsa_dict[site_index][quantity_index]) != 0
                ):
                    data_file_name = (
                        directory
                        + site_index_string
                        + self.prefix
                        + "_"
                        + self.data_titles[quantity_index]
                        + ".txt"
                    )
                    with open(data_file_name, "w") as data_file:
                        data_file.writelines(
                            "%s\n" % item
                            for item in self.hsa_dict[site_index][quantity_index]
                        )

    @utils.function_timer
    def write_energy_ww_breakdown(self):
        number_of_shells = len(self.energy_ww_lr_breakdown[0])
        if self.energy_ww_lr_breakdown is not None:
            with open(self.prefix + "_energy_ww_by_shell.txt", "w") as output_file:
                output_file.write("index ")
                for shell_index in range(number_of_shells):
                    output_file.write("shell_%d " % shell_index)
                output_file.write("\n")
                for site_index in range(self.hsa_data.shape[0]):
                    output_file.write("%d " % site_index)
                    for shell in self.energy_ww_lr_breakdown[site_index]:
                        output_file.write("%6.3f " % shell)
                    output_file.write("\n")
                output_file.write("\n")

    def write_angular_structure_distribution(self):
        directory = self.prefix + "_hsa_data"
        if not os.path.exists(directory):
            os.makedirs(directory)
        if self.angular_st_distribution is not None:
            for site_index in range(self.hsa_data.shape[0]):
                site_index_string = "/%03d_" % site_index
                data_file_name = (
                    directory + site_index_string + self.prefix + "_r_theta.txt"
                )
                with open(data_file_name, "w") as data_file:
                    lines = [
                        "{0[0]:.3f} {0[1]:.3f}\n".format(item)
                        for item in self.angular_st_distribution[site_index]
                    ]
                    data_file.writelines(lines)
