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
import sys
import typing

# custom
import numpy
import parmed
import mdtraj

# local
from sstmap import utils


## constants

DON_ACC_LIST = ["oxygen", "nitrogen", "sulfur"]

_WATER_RESNAMES = [
    "H2O",
    "HHO",
    "OHH",
    "HOH",
    "OH2",
    "SOL",
    "WAT",
    "TIP",
    "TIP2",
    "TIP3",
    "TIP4",
    "T3P",
    "T4P",
    "T5P",
]

ANGLE_CUTOFF_RAD = 0.523599

REQUIREMENTS: dict[str, list[str]] = {
    "prmtop": ["prmtop", "", "lorentz-bertholot"],
    "parm7": ["parm7", "", "lorentz-bertholot"],
    "psf": [
        "toppar",
        "Please provide a folder named toppar that contains charmm parameter/topology files.",
        "lorentz-bertholot",
    ],
    "gro": [
        "top",
        "Please provide graomcs .top file corresponding to your system and also make sure that .itp files "
        "are present in the directory where calculations are being run. To get a list of .itp files being "
        "used by gromacs topology file, type $grep #include ",
        "lorentz-bertholot",
    ],
    "pdb": [
        "txt",
        "Please provide a text file containing non-bonded parameters for your system.",
        "geometric",
    ],
    "h5": [
        "txth5",
        "Please provide a text file containing non-bonded parameters for your system.",
        "lorentz-bertholot",
    ],
}


## classes


class WaterAnalysis(object):
    def __init__(
        self,
        topology_file: str,
        trajectory: str,
        supporting_file: typing.Optional[str] = None,
    ):
        if not os.path.exists(topology_file) or not os.path.exists(trajectory):
            raise IOError("File %s or %s does not exist." % (topology_file, trajectory))
        self.topology_file = topology_file
        self.trajectory = trajectory
        self.supporting_file = supporting_file
        topology_extension = self.topology_file.split(".")[-1]
        required_support = REQUIREMENTS[topology_extension][0]
        self.comb_rule: typing.Optional[str] = None

        if required_support == topology_extension:
            self.supporting_file = self.topology_file
            self.comb_rule = REQUIREMENTS[topology_extension][-1]
        else:
            if topology_extension not in list(REQUIREMENTS.keys()):
                message = (
                    """SSTMap currently does not support %s topology file type.
                If this is a non-standard force-filed, consider using a PDB file as a topplogy
                and provide a text file containing non-bonded parameters for each atom in your system.
                See sstmap.org for more details.
                """
                    % topology_extension
                )
                sys.exit(message)
            else:
                self.supporting_file = supporting_file
                self.comb_rule = REQUIREMENTS[topology_extension][-1]

        if self.topology_file.endswith(".h5"):
            print("topology ends with h5")
            first_frame = mdtraj.load_frame(self.trajectory, 0)
        else:
            first_frame = mdtraj.load_frame(self.trajectory, 0, top=self.topology_file)
        assert (
            first_frame.unitcell_lengths is not None
        ), "Could not detect unit cell information."
        self.topology = first_frame.topology

        super_wat_select_exp = ""
        for index, wat_res in enumerate(_WATER_RESNAMES):
            if index < len(_WATER_RESNAMES) - 1:
                super_wat_select_exp += "resname %s or " % wat_res
            else:
                super_wat_select_exp += "resname %s" % wat_res

        self.all_atom_ids = self.topology.select("all")
        self.prot_atom_ids = self.topology.select("protein")
        self.wat_atom_ids = self.topology.select("water")
        self.set_neighbors("water and name O")

        if self.wat_atom_ids.shape[0] == 0:
            self.wat_atom_ids = self.topology.select(super_wat_select_exp)
        assert (
            self.wat_atom_ids.shape[0] != 0
        ), "Unable to recognize water residues in the system!"
        assert (
            self.topology.atom(self.wat_atom_ids[0]).name == "O"
        ), "Failed while constructing water oxygen atom indices!"

        self.wat_oxygen_atom_ids = numpy.asarray(
            [atom for atom in self.wat_atom_ids if self.topology.atom(atom).name == "O"]
        )
        self.water_sites = self.wat_oxygen_atom_ids[1] - self.wat_oxygen_atom_ids[0]

        for oxygen_index in self.wat_oxygen_atom_ids:
            oxygen_name = self.topology.atom(oxygen_index).name[0]
            hydrogen1_name = self.topology.atom(oxygen_index + 1).name[0]
            hydrogen2_name = self.topology.atom(oxygen_index + 2).name[0]
            if oxygen_name != "O" or hydrogen1_name != "H" or hydrogen2_name != "H":
                sys.exit(
                    "Water molecules in the topology must be organized as Oxygen, Hydrogen, Hydrogen, Virtual-sites."
                )

        self.non_water_atom_ids = numpy.setdiff1d(self.all_atom_ids, self.wat_atom_ids)
        self.non_prot_atom_ids = numpy.setdiff1d(
            self.non_water_atom_ids, self.prot_atom_ids
        )

        if self.prot_atom_ids.shape[0] == 0:
            self.prot_atom_ids = self.non_water_atom_ids
        assert (
            self.wat_atom_ids.shape[0] + self.non_water_atom_ids.shape[0]
            == self.all_atom_ids.shape[0]
        ), "Failed to partition atom indices in the system correctly!"

        print("Obtaining non-bonded parameters for the system ...")
        self.chg_product, self.acoeff, self.bcoeff = self.generate_nonbonded_params()
        assert (
            self.chg_product.shape == self.acoeff.shape == self.bcoeff.shape
        ), "Mismatch in non-bonded parameter matrices, exiting."
        print("Done.")

        print("Assigning hydrogen bond types ...")
        self.don_H_pair_dict: dict[int, list[list[int]]] = {}
        self.prot_hb_types = numpy.zeros(len(self.all_atom_ids), dtype=int)
        self.solute_acc_ids, self.solute_don_ids, self.solute_acc_don_ids = (
            self.assign_hb_types()
        )
        print("Done.")

    def set_neighbors(self, mask: str):
        self.neighbor_ids = self.topology.select(mask)
        self.wat_nbrs_shell = numpy.zeros(self.neighbor_ids.shape[0], dtype=int)

    @utils.function_timer
    def assign_hb_types(
        self,
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        self.topology.create_standard_bonds()
        acceptor_list: list[int] = []
        donor_list: list[int] = []
        acceptor_donor_list: list[int] = []

        non_water_bonds = [
            (bond[0].index, bond[1].index)
            for bond in self.topology.bonds
            if bond[0].residue.name not in _WATER_RESNAMES
        ]
        distance_pairs: list[list[int]] = []
        keys_all: list[int] = []

        for atom_index in self.prot_atom_ids:
            if self.topology.atom(atom_index).element.name in DON_ACC_LIST:
                bonds_of_atom: list[tuple[int, int]] = []
                for bond in non_water_bonds:
                    if atom_index in bond and bond not in bonds_of_atom:
                        bonds_of_atom.append(bond)

                if self.topology.atom(atom_index).element.name == "nitrogen":
                    donor_hydrogen_pairs: list[list[int]] = []
                    for atom1, atom2 in bonds_of_atom:
                        if self.topology.atom(atom2).element.name == "hydrogen":
                            donor_hydrogen_pairs.append([atom1, atom2])
                        if self.topology.atom(atom1).element.name == "hydrogen":
                            donor_hydrogen_pairs.append([atom2, atom1])
                    if len(donor_hydrogen_pairs) != 0:
                        keys_all.append(atom_index)
                        for bond in donor_hydrogen_pairs:
                            distance_pairs.append(bond)
                        if atom_index not in donor_list:
                            donor_list.append(atom_index)
                    else:
                        acceptor_list.append(atom_index)

                if self.topology.atom(atom_index).element.name in [
                    "oxygen",
                    "sulfur",
                ]:
                    donor_hydrogen_pairs = []
                    for atom1, atom2 in bonds_of_atom:
                        if self.topology.atom(atom2).element.name == "hydrogen":
                            donor_hydrogen_pairs.append([atom1, atom2])
                        if self.topology.atom(atom1).element.name == "hydrogen":
                            donor_hydrogen_pairs.append([atom2, atom1])
                    if len(donor_hydrogen_pairs) != 0:
                        keys_all.append(atom_index)
                        for bond in donor_hydrogen_pairs:
                            distance_pairs.append(bond)
                        if atom_index not in acceptor_donor_list:
                            acceptor_donor_list.append(atom_index)
                    else:
                        acceptor_list.append(atom_index)

        for _index, pair in enumerate(distance_pairs):
            if pair[0] not in list(self.don_H_pair_dict.keys()):
                self.don_H_pair_dict[pair[0]] = [[pair[0], pair[1]]]
            else:
                self.don_H_pair_dict[pair[0]].append([pair[0], pair[1]])

        solute_acceptor_ids = numpy.array(acceptor_list, dtype=int)
        solute_acceptor_donor_ids = numpy.array(acceptor_donor_list, dtype=int)
        solute_donor_ids = numpy.array(donor_list, dtype=int)

        for atom_id in solute_acceptor_ids:
            self.prot_hb_types[atom_id] = 1
        for atom_id in solute_donor_ids:
            self.prot_hb_types[atom_id] = 2
        for atom_id in solute_acceptor_donor_ids:
            self.prot_hb_types[atom_id] = 3

        return solute_acceptor_ids, solute_donor_ids, solute_acceptor_donor_ids

    @utils.function_timer
    def generate_nonbonded_params(
        self,
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        vdw: list[typing.Any] = []
        chg: list[float] = []

        if self.supporting_file is None:
            raise ValueError(
                "No supporting file provided. Please provide a supporting file "
                "containing non-bonded parameters for your system."
            )

        if self.supporting_file.endswith(".txt"):
            nb_data = numpy.loadtxt(self.supporting_file)
            for charge in nb_data[:, 0]:
                chg.append(charge)
            for vdw_params in nb_data[:, 1:]:
                vdw.append(vdw_params)
            chg_array = numpy.asarray(chg)

        elif self.supporting_file.endswith(".txth5"):
            nb_data = numpy.loadtxt(self.supporting_file)
            for charge in nb_data[:, 0]:
                chg.append(charge)
            for vdw_params in nb_data[:, 1:]:
                vdw.append(vdw_params)
            chg_array = numpy.asarray(chg)

        elif self.topology_file.endswith(".psf"):
            parmed_topology_object = parmed.load_file(self.topology_file)
            param_dir = os.path.abspath(self.supporting_file)
            param_files = [
                os.path.join(param_dir, filename)
                for filename in os.listdir(param_dir)
                if os.path.isfile(os.path.join(param_dir, filename))
                and filename.endswith((".rtf", ".top", ".par", ".prm", ".inp", ".str"))
            ]
            params = parmed.charmm.CharmmParameterSet(*param_files)
            try:
                parmed_topology_object.load_parameters(params)
            except Exception as exception:
                print(exception)
            for atom_index in self.all_atom_ids:
                vdw.append(
                    [
                        parmed_topology_object.atoms[atom_index].sigma,
                        parmed_topology_object.atoms[atom_index].epsilon,
                    ]
                )
                chg.append(parmed_topology_object.atoms[atom_index].charge)
            chg_array = numpy.asarray(chg) * 18.2223

        else:
            parmed_topology_object = parmed.load_file(self.supporting_file)
            for atom_index in self.all_atom_ids:
                vdw.append(
                    [
                        parmed_topology_object.atoms[atom_index].sigma,
                        parmed_topology_object.atoms[atom_index].epsilon,
                    ]
                )
                chg.append(parmed_topology_object.atoms[atom_index].charge)
            chg_array = numpy.asarray(chg) * 18.2223

        vdw_array = numpy.asarray(vdw)
        water_chg = chg_array[self.wat_atom_ids[0 : self.water_sites]].reshape(
            self.water_sites, 1
        )
        chg_product = water_chg * numpy.tile(
            chg_array[self.all_atom_ids], (self.water_sites, 1)
        )

        water_sig = vdw_array[self.wat_atom_ids[0 : self.water_sites], 0].reshape(
            self.water_sites, 1
        )
        water_eps = vdw_array[self.wat_atom_ids[0 : self.water_sites], 1].reshape(
            self.water_sites, 1
        )

        mixed_sig: typing.Optional[numpy.ndarray] = None
        mixed_eps: typing.Optional[numpy.ndarray] = None

        if self.comb_rule is None or self.comb_rule == "lorentz-bertholot":
            mixed_sig = 0.5 * (water_sig + vdw_array[self.all_atom_ids, 0])
            mixed_eps = numpy.sqrt(water_eps * vdw_array[self.all_atom_ids, 1])
        if self.comb_rule == "geometric":
            mixed_sig = numpy.sqrt(water_sig * vdw_array[self.all_atom_ids, 0])
            mixed_eps = numpy.sqrt(water_eps * vdw_array[self.all_atom_ids, 1])

        if mixed_eps is not None and mixed_sig is not None:
            acoeff = 4 * mixed_eps * (mixed_sig**12)
            bcoeff = 4 * mixed_eps * (mixed_sig**6)
        else:
            raise Exception("Couldn't assign vdw params")

        return chg_product, acoeff, bcoeff

    def calculate_hydrogen_bonds(
        self,
        trajectory: mdtraj.Trajectory,
        water: int,
        neighbors: numpy.ndarray,
        water_water: bool = True,
    ) -> numpy.ndarray:
        hbond_data: list[typing.Any] = []
        angle_triplets: list[list[int]] = []

        if water_water:
            for water_neighbor in neighbors:
                angle_triplets.extend(
                    [
                        [water, water_neighbor, water_neighbor + 1],
                        [water, water_neighbor, water_neighbor + 2],
                        [water_neighbor, water, water + 1],
                        [water_neighbor, water, water + 2],
                    ]
                )
        else:
            for solute_neighbor in neighbors:
                if (
                    self.prot_hb_types[solute_neighbor] == 1
                    or self.prot_hb_types[solute_neighbor] == 3
                ):
                    angle_triplets.extend(
                        [
                            [solute_neighbor, water, water + 1],
                            [solute_neighbor, water, water + 2],
                        ]
                    )
                if (
                    self.prot_hb_types[solute_neighbor] == 2
                    or self.prot_hb_types[solute_neighbor] == 3
                ):
                    for donor_hydrogen_pair in self.don_H_pair_dict[solute_neighbor]:
                        angle_triplets.extend(
                            [[water, solute_neighbor, donor_hydrogen_pair[1]]]
                        )

        angle_triplets_array = numpy.asarray(angle_triplets)
        angles = mdtraj.compute_angles(trajectory, angle_triplets_array)
        angles[numpy.isnan(angles)] = 0.0
        hbonds = angle_triplets_array[numpy.where(angles[0, :] <= ANGLE_CUTOFF_RAD)]
        return hbonds

    def water_nbr_orientations(
        self,
        trajectory: mdtraj.Trajectory,
        water: int,
        neighbors: numpy.ndarray,
    ) -> list[float]:
        angle_triplets: list[list[int]] = []

        for water_neighbor in neighbors:
            angle_triplets.extend(
                [
                    [water, water_neighbor, water_neighbor + 1],
                    [water, water_neighbor, water_neighbor + 2],
                    [water_neighbor, water, water + 1],
                    [water_neighbor, water, water + 2],
                ]
            )

        angle_triplets_array = numpy.asarray(angle_triplets)
        angles = mdtraj.compute_angles(trajectory, angle_triplets_array)
        angles[numpy.isnan(angles)] = 0.0

        water_orientations = [
            numpy.rad2deg(numpy.min(angles[0, index * 4 : (index * 4) + 4]))
            for index in range(neighbors.shape[0])
        ]
        return water_orientations
