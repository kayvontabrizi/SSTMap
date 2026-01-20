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
import time
import functools
import typing

# custom
import numpy
import scipy.stats
import matplotlib
import matplotlib.pyplot
import matplotlib.ticker
import matplotlib.cm

matplotlib.use("Agg")


## methods


def function_timer(
    function: typing.Callable[..., typing.Any],
) -> typing.Callable[..., typing.Any]:
    @functools.wraps(function)
    def timed_function(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        start_time = time.time()
        result = function(*args, **kwargs)
        end_time = time.time()
        print(
            "Total time running %s: %2.2f seconds"
            % (function.__name__, end_time - start_time)
        )
        return result

    return timed_function


def print_progress_bar(count: int, total: int):
    bar_length = 20
    filled_length = int(round(bar_length * count / float(total)))
    percentage = round(100.0 * count / float(total), 1)
    bar = "=" * filled_length + " " * (bar_length - filled_length)
    sys.stdout.write("Progress |%s| %s%s Done.\r" % (bar, percentage, "%"))
    sys.stdout.flush()
    if count == total:
        print()


def plot_enbr(
    data_directory: str,
    site_indices: typing.Optional[list[int]] = None,
    neighbor_normalization: bool = False,
    reference_data: typing.Optional[str] = None,
    reference_neighbors: typing.Optional[float] = None,
):
    energy_neighbor_files: list[str] = []
    energy_neighbor_data: dict[int, numpy.ndarray] = {}
    reference_energy_neighbor: typing.Optional[numpy.ndarray] = None
    neighbor_files: list[str] = []
    neighbor_values: list[float] = []

    if not os.path.isdir(data_directory):
        sys.exit("Data directory not found, please check path of the directory again.")

    if site_indices is None:
        energy_neighbor_files = [
            filename
            for filename in os.listdir(data_directory)
            if filename.endswith("Ewwnbr.txt")
        ]
        if neighbor_normalization:
            neighbor_files = [
                filename
                for filename in os.listdir(data_directory)
                if filename.endswith("Nnbrs.txt")
            ]
    else:
        energy_neighbor_files = [
            filename
            for filename in os.listdir(data_directory)
            if filename.endswith("Ewwnbr.txt") and int(filename[0:3]) in site_indices
        ]
        if neighbor_normalization:
            neighbor_files = [
                filename
                for filename in os.listdir(data_directory)
                if filename.endswith("Nnbrs.txt") and int(filename[0:3]) in site_indices
            ]

    for index, filename in enumerate(energy_neighbor_files):
        site_index = int(filename[0:3])
        energy_neighbor_data[site_index] = numpy.loadtxt(
            data_directory + "/" + filename
        )
        if neighbor_normalization:
            neighbors = numpy.loadtxt(data_directory + "/" + neighbor_files[index])
            neighbor_values.append(numpy.sum(neighbors) / neighbors.shape[0])

    if reference_data is not None:
        reference_energy_neighbor = numpy.loadtxt(reference_data)
        if neighbor_normalization and reference_neighbors is not None:
            reference_energy_neighbor *= reference_neighbors

    for index, site_index in enumerate(energy_neighbor_data.keys()):
        print(("Generating Enbr plot for: ", site_index, energy_neighbor_files[index]))
        site_energy_neighbor = energy_neighbor_data[site_index] * 0.5
        x_low, x_high = -5.0, 3.0
        energy_min, energy_max = (
            numpy.min(site_energy_neighbor),
            numpy.max(site_energy_neighbor),
        )
        if energy_min < x_low:
            x_low = energy_min
        if energy_max > x_high:
            x_high = energy_max

        x_values = numpy.linspace(x_low, x_high)
        kernel = scipy.stats.gaussian_kde(site_energy_neighbor)
        probability_x = kernel.evaluate(x_values)
        if neighbor_normalization:
            site_neighbors = neighbor_values[index]
            probability_x *= site_neighbors

        probability_x_reference: typing.Optional[numpy.ndarray] = None
        if reference_energy_neighbor is not None:
            kernel = scipy.stats.gaussian_kde(reference_energy_neighbor)
            probability_x_reference = kernel.evaluate(x_values)

        figure, axes = matplotlib.pyplot.subplots(1)
        figure.set_size_inches(3, 3)
        matplotlib.pyplot.xlim(x_low, x_high)
        matplotlib.pyplot.ylim(0.0, numpy.max(probability_x) + 0.1)
        start, end = axes.get_ylim()
        axes.yaxis.set_ticks(numpy.arange(start, end, 0.2))
        start, end = axes.get_xlim()
        axes.xaxis.set_ticks(numpy.arange(start, end, 2.0))
        axes.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%0.1f"))
        axes.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%0.1f"))
        x_label = r"$\mathit{E_{n} (kcal/mol)}$"
        y_label = r"$\mathit{\rho(E_{n})}$"
        if neighbor_normalization:
            y_label = r"$\mathit{\rho(E_{n})N^{nbr}}$"
        axes.set_xlabel(x_label, size=14)
        axes.set_ylabel(y_label, size=14)
        axes.yaxis.tick_left()
        axes.xaxis.tick_bottom()
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)
        matplotlib.pyplot.minorticks_on()
        matplotlib.pyplot.tick_params(which="major", width=1, length=4, direction="in")
        matplotlib.pyplot.tick_params(which="minor", width=1, length=2, direction="in")
        matplotlib.pyplot.tick_params(axis="x", labelsize=12)
        matplotlib.pyplot.tick_params(axis="y", labelsize=12)
        matplotlib.pyplot.plot(
            x_values,
            probability_x,
            antialiased=True,
            linewidth=1.0,
            color="red",
            label=site_index,
        )
        if probability_x_reference is not None:
            matplotlib.pyplot.plot(
                x_values,
                probability_x_reference,
                antialiased=True,
                linewidth=1.0,
                color="green",
                label="Reference",
            )
        figure_name = "%03d_" % site_index
        matplotlib.pyplot.legend(loc="upper right", prop={"size": 10}, frameon=False)
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.savefig(
            data_directory + "/" + figure_name + "Enbr_plot.png", dpi=300
        )
        matplotlib.pyplot.close()


def plot_rtheta(
    data_directory: str,
    site_indices: typing.Optional[list[int]] = None,
):
    rtheta_files: list[str] = []
    rtheta_data: dict[int, numpy.ndarray] = {}

    print(data_directory)
    if not os.path.isdir(data_directory):
        sys.exit("Data directory not found, please check path of the directory again.")

    if site_indices is None:
        rtheta_files = [
            filename
            for filename in os.listdir(data_directory)
            if filename.endswith("r_theta.txt")
        ]
    else:
        rtheta_files = [
            filename
            for filename in os.listdir(data_directory)
            if filename.endswith("r_theta.txt") and int(filename[0:3]) in site_indices
        ]

    for index, filename in enumerate(rtheta_files):
        site_index = int(filename[0:3])
        rtheta_data[site_index] = numpy.loadtxt(data_directory + "/" + filename)

    integration_counts = 16.3624445886
    for index, site_index in enumerate(rtheta_data.keys()):
        print(("Generating r_theta plot for: ", site_index, rtheta_files[index]))
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(projection="3d")
        theta = rtheta_data[site_index][:, 0]
        radius = rtheta_data[site_index][:, 1]

        x_mesh, y_mesh = numpy.mgrid[0:130:131j, 2.0:6.0:41j]
        values = numpy.vstack([theta, radius])
        kernel = scipy.stats.gaussian_kde(values)
        positions = numpy.vstack([x_mesh.ravel(), y_mesh.ravel()])
        z_mesh = numpy.reshape(kernel(positions).T, x_mesh.shape)
        z_mesh *= integration_counts * 0.1

        sum_counts_kernel = 0
        for mesh_index in range(0, y_mesh.shape[1]):
            distance = y_mesh[0, mesh_index]
            distance_low = distance - 0.1
            volume = (4.0 / 3.0) * numpy.pi * (distance**3)
            volume_low = (4.0 / 3.0) * numpy.pi * (distance_low**3)
            shell_volume = volume - volume_low
            counts_bulk = 0.0329 * shell_volume
            sum_counts_kernel += numpy.sum(z_mesh[:, mesh_index])
            z_mesh[:, mesh_index] = z_mesh[:, mesh_index], counts_bulk

        print(sum_counts_kernel)
        legend_label = "%03d_" % site_index
        axes.plot_surface(
            x_mesh,
            y_mesh,
            z_mesh,
            rstride=1,
            cstride=1,
            linewidth=0.5,
            antialiased=True,
            alpha=1.0,
            cmap=matplotlib.cm.coolwarm,
            label=legend_label,
        )
        x_label = r"$\theta^\circ$"
        y_label = r"$r (\AA)$"
        axes.set_xlabel(x_label)
        axes.set_xlim(0, 130)
        axes.set_ylabel(y_label)
        axes.set_ylim(2.0, 6.0)
        z_label = r"$\mathrm{P(\theta, \AA)}$"
        axes.set_zlabel(z_label)
        matplotlib.pyplot.savefig(
            data_directory + "/" + legend_label + "rtheta_plot.png", dpi=300
        )
        matplotlib.pyplot.close()


def read_hsa_summary(hsa_data_file: str) -> dict[int, list[float]]:
    file_handle = open(hsa_data_file, "r")
    data = file_handle.readlines()
    hsa_header = data[0]
    data_keys = hsa_header.strip("\n").split()
    hsa_data: dict[int, list[float]] = {}
    for line in data[1:]:
        float_converted_data = [float(x) for x in line.strip("\n").split()[1:27]]
        hsa_data[int(line.strip("\n").split()[0])] = float_converted_data
    file_handle.close()
    return hsa_data


def read_gist_summary(gist_data_file: str) -> dict[int, list[float]]:
    file_handle = open(gist_data_file, "r")
    data = file_handle.readlines()
    hsa_header = data[0]
    data_keys = hsa_header.strip("\n").split()
    hsa_data: dict[int, list[float]] = {}
    for line in data[1:]:
        float_converted_data = [float(x) for x in line.strip("\n").split()[1:27]]
        hsa_data[int(line.strip("\n").split()[0])] = float_converted_data
    file_handle.close()
    return hsa_data


def write_watpdb_from_list(
    coordinates: numpy.ndarray,
    filename: str,
    water_id_list: list[tuple[int, int]],
    full_water_residue: bool = False,
):
    pdb_line_format = "{0:6}{1:>5}  {2:<3}{3:<1}{4:>3} {5:1}{6:>4}{7:1}   {8[0]:>8.3f}{8[1]:>8.3f}{8[2]:>8.3f}{9:>6.2f}{10:>6.2f}{11:>12s}\n"
    ter_line_format = "{0:3}   {1:>5}      {2:>3} {3:1}{4:4} \n"
    pdb_lines: list[str] = []

    atom_number = 1
    residue_number = 1
    with open(filename + ".pdb", "w") as file_handle:
        for water_index in range(len(water_id_list)):
            water = water_id_list[water_index]
            atom_index = atom_number
            residue_index = residue_number % 10000
            water_coordinates = coordinates[water[0], water[1], :]
            chain_id = "A"
            pdb_line = pdb_line_format.format(
                "ATOM",
                atom_index,
                "O",
                " ",
                "WAT",
                chain_id,
                residue_index,
                " ",
                water_coordinates,
                0.00,
                0.00,
                "O",
            )
            file_handle.write(pdb_line)

            if full_water_residue:
                hydrogen1_coordinates = coordinates[water[0], water[1] + 1, :]
                pdb_line_hydrogen1 = pdb_line_format.format(
                    "ATOM",
                    atom_index + 1,
                    "H1",
                    " ",
                    "WAT",
                    chain_id,
                    residue_index,
                    " ",
                    hydrogen1_coordinates,
                    0.00,
                    0.00,
                    "H",
                )
                file_handle.write(pdb_line_hydrogen1)
                hydrogen2_coordinates = coordinates[water[0], water[1] + 2, :]
                pdb_line_hydrogen2 = pdb_line_format.format(
                    "ATOM",
                    atom_index + 2,
                    "H2",
                    " ",
                    "WAT",
                    chain_id,
                    residue_index,
                    " ",
                    hydrogen2_coordinates,
                    0.00,
                    0.00,
                    "H",
                )
                file_handle.write(pdb_line_hydrogen2)
                atom_number += 3
                residue_number += 1
            else:
                atom_number += 1
                residue_number += 1
            if residue_index == 9999:
                ter_line = ter_line_format.format(
                    "TER", atom_number, "WAT", chain_id, residue_index
                )
                atom_number = 1


def write_watpdb_from_coords(
    filename: str,
    coordinates: numpy.ndarray,
    full_water_residue: bool = False,
):
    pdb_line_format = "{0:6}{1:>5}  {2:<3}{3:<1}{4:>3} {5:1}{6:>4}{7:1}   {8[0]:>8.3f}{8[1]:>8.3f}{8[2]:>8.3f}{9:>6.2f}{10:>6.2f}{11:>12s}\n"
    ter_line_format = "{0:3}   {1:>5}      {2:>3} {3:1}{4:4} \n"
    pdb_lines: list[str] = []

    atom_number = 0
    residue_number = 0
    water_index = 0
    with open(filename + ".pdb", "w") as file_handle:
        file_handle.write("REMARK Initial number of clusters: N/A\n")
        while water_index < len(coordinates):
            atom_index = atom_number
            residue_index = residue_number % 10000
            water_coordinates = coordinates[water_index]
            chain_id = "A"
            pdb_line = pdb_line_format.format(
                "ATOM",
                atom_index,
                "O",
                " ",
                "WAT",
                chain_id,
                residue_index,
                " ",
                water_coordinates,
                0.00,
                0.00,
                "O",
            )
            file_handle.write(pdb_line)
            water_index += 1
            if full_water_residue:
                hydrogen1_coordinates = coordinates[water_index]
                pdb_line_hydrogen1 = pdb_line_format.format(
                    "ATOM",
                    atom_index + 1,
                    "H1",
                    " ",
                    "WAT",
                    chain_id,
                    residue_index,
                    " ",
                    hydrogen1_coordinates,
                    0.00,
                    0.00,
                    "H",
                )
                file_handle.write(pdb_line_hydrogen1)
                hydrogen2_coordinates = coordinates[water_index + 1]
                pdb_line_hydrogen2 = pdb_line_format.format(
                    "ATOM",
                    atom_index + 2,
                    "H2",
                    " ",
                    "WAT",
                    chain_id,
                    residue_index,
                    " ",
                    hydrogen2_coordinates,
                    0.00,
                    0.00,
                    "H",
                )
                file_handle.write(pdb_line_hydrogen2)
                atom_number += 3
                residue_number += 1
                water_index += 2
            else:
                atom_number += 1
                residue_number += 1
            if residue_index == 9999:
                ter_line = ter_line_format.format(
                    "TER", atom_number, "WAT", chain_id, residue_index
                )
                atom_number = 1


## classes


class GISTFields:
    data_titles = [
        "index",
        "x",
        "y",
        "z",
        "N_wat",
        "g_O",
        "g_H",
        "TS_tr_dens",
        "TS_tr_norm",
        "TS_or_dens",
        "TS_or_norm",
        "dTSsix-dens",
        "dTSsix_norm",
        "E_sw_dens",
        "E_sw_norm",
        "E_ww_dens",
        "Eww_norm",
        "E_ww_nbr_dens",
        "E_ww_nbr_norm",
        "N_nbr_dens",
        "N_nbr_norm",
        "f_hb_dens",
        "f_hb_norm",
        "N_hb_sw_dens",
        "N_hb_sw_norm",
        "N_hb_ww_dens",
        "N_hb_ww_norm",
        "N_don_sw_dens",
        "N_don_sw_norm",
        "N_acc_sw_dens",
        "N_acc_sw_norm",
        "N_don_ww_dens",
        "N_don_ww_norm",
        "N_acc_ww_dens",
        "N_acc_ww_norm",
    ]
    index = 0
    x = 1
    y = 2
    z = 3
    N_wat = 4
    g_O = 5
    g_H = 6
    TS_tr_dens = 7
    TS_tr_norm = 8
    TS_or_dens = 9
    TS_or_norm = 10
    dTSsix_dens = 11
    dTSsix_norm = 12
    E_sw_dens = 13
    E_sw_norm = 14
    E_ww_dens = 15
    Eww_norm = 16
    E_ww_nbr_dens = 17
    E_ww_nbr_norm = 18
    N_nbr_dens = 19
    N_nbr_norm = 20
    f_hb_dens = 21
    f_hb_norm = 22
    N_hb_sw_dens = 23
    N_hb_sw_norm = 24
    N_hb_ww_dens = 25
    N_hb_ww_norm = 26
    N_don_sw_dens = 27
    N_don_sw_norm = 28
    N_acc_sw_dens = 29
    N_acc_sw_norm = 30
    N_don_ww_dens = 31
    N_don_ww_norm = 32
    N_acc_ww_dens = 33
    N_acc_ww_norm = 34


class HSAFields:
    data_titles = [
        "index",
        "x",
        "y",
        "z",
        "N_wat",
        "g_O",
        "g_H",
        "TS_tr_dens",
        "TS_tr_norm",
        "TS_or_dens",
        "TS_or_norm",
        "dTSsix-dens",
        "dTSsix_norm",
        "E_sw_dens",
        "E_sw_norm",
        "E_ww_dens",
        "Eww_norm",
        "E_ww_nbr_dens",
        "E_ww_nbr_norm",
        "N_nbr_dens",
        "N_nbr_norm",
        "f_hb_dens",
        "f_hb_norm",
        "N_hb_sw_dens",
        "N_hb_sw_norm",
        "N_hb_ww_dens",
        "N_hb_ww_norm",
        "N_don_sw_dens",
        "N_don_sw_norm",
        "N_acc_sw_dens",
        "N_acc_sw_norm",
        "N_don_ww_dens",
        "N_don_ww_norm",
        "N_acc_ww_dens",
        "N_acc_ww_norm",
    ]
    index = 0
    x = 1
    y = 2
    z = 3
    nwat = 4
    occupancy = 5
    Esw = 6
    EswLJ = 7
    EswElec = 8
    Eww = 9
    EwwLJ = 10
    EwwElec = 11
    Etot = 12
    Ewwnbr = 13
    TSsw_trans = 14
    TSsw_orient = 15
    TStot = 16
    Nnbrs = 17
    Nhbww = 18
    Nhbsw = 19
    Nhbtot = 20
    f_hb_ww = 21
    f_enc = 22
    Acc_ww = 23
    Don_ww = 24
    Acc_sw = 25
    Don_sw = 26
    solute_acceptors = 27
    solute_donors = 28
