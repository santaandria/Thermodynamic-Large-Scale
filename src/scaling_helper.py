"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  ░█▀▀░█▀█░█▀█░▀█▀░█▀█░░░█▀█░█▀█░█▀▄░█▀▄░▀█▀░█▀█                              ║
║  ░▀▀█░█▀█░█░█░░█░░█▀█░░░█▀█░█░█░█░█░█▀▄░░█░░█▀█                              ║
║  ░▀▀▀░▀░▀░▀░▀░░▀░░▀░▀░░░▀░▀░▀░▀░▀▀░░▀░▀░▀▀▀░▀░▀                              ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │  Author: Santa Andria                                                   │ ║
║  │  Email:  santa.andria@dicea.unipd.it                                    │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio

import matplotlib.pyplot as plt
from src.mevpt import *

##### Plotting helper ####
KG_RASTER_PATH = Path("./data/CONUS_Beck_KG_V1_Present.tif")
KG_DESCRIPTIONS = {
    "Af": ("Af - Tropical, rainforest", [0, 0, 255], 1),
    "Am": ("Am - Tropical, monsoon", [0, 120, 255], 2),
    "Aw": ("Aw - Tropical, savannah", [70, 170, 250], 3),
    "BWh": ("BWh - Arid, desert, hot", [255, 0, 0], 4),
    "BWk": ("BWk - Arid, desert, cold", [255, 150, 150], 5),
    "BSh": ("BSh - Arid, steppe, hot", [245, 165, 0], 6),
    "BSk": ("BSk - Arid, steppe, cold", [255, 220, 100], 7),
    "Csa": ("Csa - Temperate, dry summer, hot summer", [255, 255, 0], 8),
    "Csb": ("Csb - Temperate, dry summer, warm summer", [200, 200, 0], 9),
    "Csc": ("Csc - Temperate, dry summer, cold summer", [150, 150, 0], 10),
    "Cwa": ("Cwa - Temperate, dry winter, hot summer", [150, 255, 150], 11),
    "Cwb": ("Cwb - Temperate, dry winter, warm summer", [100, 200, 100], 12),
    "Cwc": ("Cwc - Temperate, dry winter, cold summer", [50, 150, 50], 13),
    "Cfa": ("Cfa - Temperate, no dry season, hot summer", [200, 255, 80], 14),
    "Cfb": ("Cfb - Temperate, no dry season, warm summer", [100, 255, 80], 15),
    "Cfc": ("Cfc - Temperate, no dry season, cold summer", [50, 200, 0], 16),
    "Dsa": ("Dsa - Cold, dry summer, hot summer", [255, 0, 255], 17),
    "Dsb": ("Dsb - Cold, dry summer, warm summer", [200, 0, 200], 18),
    "Dsc": ("Dsc - Cold, dry summer, cold summer", [150, 50, 150], 19),
    "Dsd": ("Dsd - Cold, dry summer, very cold winter", [150, 100, 150], 20),
    "Dwa": ("Dwa - Cold, dry winter, hot summer", [170, 175, 255], 21),
    "Dwb": ("Dwb - Cold, dry winter, warm summer", [90, 120, 220], 22),
    "Dwc": ("Dwc - Cold, dry winter, cold summer", [75, 80, 180], 23),
    "Dwd": ("Dwd - Cold, dry winter, very cold winter", [50, 0, 135], 24),
    "Dfa": ("Dfa - Cold, no dry season, hot summer", [0, 255, 255], 25),
    "Dfb": ("Dfb - Cold, no dry season, warm summer", [55, 200, 255], 26),
    "Dfc": ("Dfc - Cold, no dry season, cold summer", [0, 125, 125], 27),
    "Dfd": ("Dfd - Cold, no dry season, very cold winter", [0, 70, 95], 28),
    "ET": ("ET - Polar, tundra", [178, 178, 178], 29),
    "EF": ("EF - Polar, frost", [102, 102, 102], 30),
}
COLORS = ["#0C5DA5", "#FF2C00", "#FF9500", "#474747", "#9e9e9e"]
MARKERS = ["o", "s", "^", "*", "+"]


def plot_group_data(
    axs, group, col_idx, var, station_list, oe_dir, ntimes=0, n_bins=20
):
    """Plot data for a single group of regions."""
    lines = []
    for d in ["1h", "24h"]:
        for j, region in enumerate(group):
            stations = station_list[station_list["KG"] == region]["StnID"].values
            prcp, exg = pool_pt(stations, d, var, oe_dir)
            pooled_cw = scaling_analysis(
                prcp,
                exg,
                var,
                ci=True if ntimes > 0 else False,
                ntimes=ntimes,
                n_bins=n_bins,
            )

            for param_idx, param in enumerate(["c", "w"]):
                ax = axs[param_idx, col_idx]

                l = ax.plot(
                    pooled_cw[f"{var}_med"],
                    pooled_cw[param],
                    "-" if d == "1h" else "--",
                    lw=1.5,
                    c="#{:02x}{:02x}{:02x}".format(*KG_DESCRIPTIONS[region][1]),
                )

                if ntimes > 0:
                    ax.fill_between(
                        pooled_cw[f"{var}_med"],
                        pooled_cw[f"{param}_up"],
                        pooled_cw[f"{param}_low"],
                        alpha=0.3,
                        color="#{:02x}{:02x}{:02x}".format(*KG_DESCRIPTIONS[region][1]),
                        edgecolor="None",
                    )

                if d == "1h" and param == "c":
                    lines.append(l[0])
    return lines


def plot_scaling(var, xlabel, station_list, oe_dir, ntimes=0, savepath=None, n_bins=20):
    ylabels = [
        (
            "$log~c~[\\text{mm} \cdot \\text{hr}^{-1}]$"
            if var == "TMEAN" or var == "DPT"
            else "$c~[\\text{mm} \cdot \\text{hr}^{-1}]$"
        ),
        "$w$",
    ]
    # groups = [['Cfa', 'Dfa', 'BSk'], ['Dfb', 'BWk', 'Csb']]
    groups = [["BSk", "BWk"], ["Cfa", "Csb"], ["Dfa", "Dfb"]]
    axtitle = ["Arid", "Temperate", "Continental"]
    params = ["c", "w"]

    fig, axs = plt.subplots(2, 3, figsize=(8, 4), constrained_layout=True)

    xlim, ylim = {param: [] for param in params}, {param: [] for param in params}
    lines = []
    for i, group in enumerate(groups):
        l = plot_group_data(
            axs, group, i, var, station_list, oe_dir, ntimes=ntimes, n_bins=n_bins
        )
        lines.extend(l)
        collect_axis_limits(axs, xlim, ylim, i)

    # Set axis properties
    setup_axes(axs, xlim, ylim, ylabels, xlabel, var, axtitle)

    # Legend
    fig.legend(
        handles=lines,
        labels=["BSk", "BWk", "Cfa", "Csb", "Dfa", "Dfb"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=6,
        borderaxespad=0.1,
    )

    if savepath:
        fig.savefig(savepath, dpi=300, format="pdf")
    plt.show()


def get_region_color(regions):
    values, colors = [], []
    for region in regions:
        rgb, value = KG_DESCRIPTIONS[region][1:]
        colors.append([c / 255.0 for c in rgb])
        values.append(value)
    return values, colors


def plot_inset(regions, main_ax, position):
    proj_5070 = ccrs.AlbersEqualArea(
        central_longitude=-96,
        central_latitude=23,
        false_easting=0.0,
        false_northing=0.0,
        standard_parallels=(29.5, 45.5),
    )
    extent = [-125, -66.75, 24.5, 49.5]
    values, colors = get_region_color(regions)

    with rasterio.open(KG_RASTER_PATH) as src:
        data = src.read(1)
        combined_mask = np.zeros_like(data, dtype=bool)
        for value in values:
            combined_mask |= data == value
        data_masked = np.ma.masked_where(~combined_mask, data)

        ax = main_ax.inset_axes(
            position, projection=proj_5070
        )  # left, bottom, width, height
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        sorted_pairs = sorted(zip(values, colors))
        sorted_values, sorted_colors = zip(*sorted_pairs)
        cmap = ListedColormap(sorted_colors)
        boundaries = list(sorted_values) + [
            max(sorted_values) + 1
        ]  # Add upper boundary
        norm = BoundaryNorm(boundaries, len(sorted_colors))

        im = ax.imshow(
            data_masked,
            cmap=cmap,
            norm=norm,
            extent=[
                src.bounds.left,
                src.bounds.right,
                src.bounds.bottom,
                src.bounds.top,
            ],
            transform=ccrs.PlateCarree(),
            interpolation="nearest",  # Prevents pixel averaging
            resample=False,
        )

        bnd = gpd.read_file("data/shp/CONUS_boundary.shp")
        ax.add_geometries(
            bnd.geometry,
            crs=ccrs.PlateCarree(),
            color="none",
            edgecolor="k",
            linewidth=0.3,
            zorder=2,
        )
    return ax


def collect_axis_limits(axs, xlim, ylim, col_idx):
    """Collect axis limits for later synchronization."""
    xlim["c"].append(axs[0, col_idx].get_xlim())
    xlim["w"].append(axs[1, col_idx].get_xlim())
    ylim["c"].append(axs[0, col_idx].get_ylim())
    ylim["w"].append(axs[1, col_idx].get_ylim())


def setup_axes(axs, xlim, ylim, ylabels, xlabel, var, axtitle):
    """Configure axis properties, labels, and limits."""
    # Convert to arrays for easier manipulation
    groups = [["BSk", "BWk"], ["Cfa", "Csb"], ["Dfa", "Dfb"]]
    for param in ["c", "w"]:
        xlim[param] = np.asarray(xlim[param])
        ylim[param] = np.asarray(ylim[param])

    for i in range(2):  # rows
        for j in range(3):  # columns
            ax = axs[i, j]
            if var in ["VIMC", "W500"]:
                ax.axvline(x=0, ls="--", c="k", lw=0.5)

            # Top row (c parameter)
            if i == 0:
                if var == "VIMC" or var == "W500":
                    pos = (
                        [0.05, 0.65, 0.35, 0.3]
                        if var == "VIMC"
                        else [0.6, 0.65, 0.35, 0.3]
                    )
                    inset = plot_inset(groups[j], ax, position=pos)
                ax.set_title(axtitle[j])
                add_cc_scaling_lines(ax, xlim, ylim, var)
                # ax.set_xlim(xlim["c"][:, 0].min(), xlim["c"][:, 1].max())
                ax.set_ylim(ylim["c"][:, 0].min(), ylim["c"][:, 1].max())

            # Bottom row (w parameter)
            if i == 1:
                if var == "TMEAN":
                    inset = plot_inset(groups[j], ax, position=[0.6, 0.65, 0.35, 0.3])
                ax.set_xlabel(xlabel)
                # ax.set_xlim(xlim["w"][:, 0].min(), xlim["w"][:, 1].max())
                ax.set_ylim(ylim["w"][:, 0].min(), ylim["w"][:, 1].max())
                ax.axhspan(ylim["w"][:, 0].min(), 1, facecolor="grey", alpha=0.25)

            # Left column
            if j == 0:
                ax.set_ylabel(ylabels[i])


def add_cc_scaling_lines(ax, xlim, ylim, var):
    """Add CC-scaling reference lines for temperature data."""
    if var == "TMEAN" or var == "DPT":
        for c0 in np.linspace(ylim["c"][:, 0].min(), ylim["c"][:, 1].max(), 3):
            T = np.linspace(xlim["c"][:, 0].min(), xlim["c"][:, 1].max(), 10)
            y_line = c0 + np.log10(0.068 + 1) * (T - np.mean(T))
            ax.plot(T, y_line, "--k", linewidth=0.2)
