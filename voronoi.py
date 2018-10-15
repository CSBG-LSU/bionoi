#!/usr/bin/env python3

from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import pandas as pd
import matplotlib
import sys,os,argparse
from biopandas.mol2 import PandasMol2
import matplotlib.pyplot as plt

# Performance tweaks
#import cProfile

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    Source
    -------
    Copied from https://gist.github.com/pv/8036995
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def fig_to_numpy(fig, alpha=1) -> np.ndarray:
    '''
    Converts matplotlib figure to a numpy array.

    Source
    ------
    Adapted from https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    '''

    # Setup figure
    fig.patch.set_alpha(alpha)
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data

def voronoi_atoms(bs, cmap, bs_out=None, size=None, alpha=0.5, save_fig=True, projection=lambda a,b: a/abs(b)**.5):
    # Suppresses warning
    pd.options.mode.chained_assignment = None

    # Read molecules in mol2 format
    mol2 = PandasMol2().read_mol2(bs)
    atoms = mol2.df[['subst_name','atom_type', 'atom_name','x','y','z']]

    # Todo
    # Align atoms to principal axis and save eigenvalue.
    # See issue #2

    # convert 3D  to 2D
    atoms["P(x)"] = atoms[['x','y','z']].apply(lambda coord: projection(coord.x,coord.z), axis=1)
    atoms["P(y)"] = atoms[['x','y','z']].apply(lambda coord: projection(coord.y,coord.z), axis=1)

    # setting output image size, labels off, set 128 dpi w x h
    size = 128 if size is None else size
    figure = plt.figure(figsize=(6 , 6),dpi=int(size))
    ax = plt.subplot(111)
    ax.axis('off')
    ax.tick_params(axis='both', bottom=False, left=False,right=False,labelleft=False, labeltop=False,labelright=False, labelbottom=False)

    # Compute Voronoi tesselation
    vor = Voronoi(atoms[['P(x)','P(y)']])
    regions, vertices = voronoi_finite_polygons_2d(vor)
    polygons = []
    for reg in regions:
        polygon = vertices[reg]
        polygons.append(polygon)
    atoms.loc[:,'polygons'] = polygons

    # Check alpha
    alpha=float(alpha)

    colors = []
    for i, row in atoms.iterrows():
        atom_type = atoms.loc[i][['atom_type']][0]
        color = cmap[atom_type]["color"]
        colors.append(color)
        colored_cell = matplotlib.patches.Polygon(row["polygons"],
                                        facecolor = color,
                                        edgecolor = 'black',
                                        alpha = alpha  )
        ax.add_patch(colored_cell)
    atoms.loc[:,"color"] = colors

    ax.set_xlim(vor.min_bound[0] , vor.max_bound[0])
    ax.set_ylim(vor.min_bound[1] , vor.max_bound[1] )

    # Output image saving in any format; default jpg
    bs_out = 'out.jpg' if bs_out is None else bs_out

    # Get image as numpy array
    figure.tight_layout(pad=0)
    img = fig_to_numpy(figure, alpha=alpha)

    if save_fig:
        plt.savefig(bs_out, frameon=False,bbox_inches="tight", pad_inches=False)

    return atoms, vor, img

def getArgs():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-mol',   default="./4v94E.mol2",      required = False, help = 'the protein/ligand mol2 file')
    parser.add_argument('-cmap',  default="./labels_mol2.csv", required = False, help = 'the cmap file. ')
    parser.add_argument('-out',   default="./out.jpg",         required = False, help = 'the outpuot image file')
    parser.add_argument('-dpi',   default="120",               required = False, help = 'image quality in dpi')
    parser.add_argument('-alpha', default="0.5",               required = False, help = 'alpha for color of cells')

    return parser.parse_args()

def Bionoi(mol, cmap, bs_out, dpi, alpha):
    # Check for color mapping file, make dict
    try:
        with open(cmap) as cMapF:
            # Parse color map file
            cmap  = np.array([line.replace("\n","").split("; ") for line in cMapF.readlines() if not line.startswith("#")])
            # To dict
            cmap = {atom:{"color":color, "definition":definition} for atom, definition, color in cmap}


    except FileNotFoundError:
        raise FileNotFoundError("Color mapping file not found in directory")

    # Run
    atoms, vor, img = voronoi_atoms(mol, cmap, bs_out=bs_out, size=dpi, alpha=alpha, save_fig=True)

    return atoms, vor, img



if __name__ == "__main__":
    args = getArgs()
    atoms, vor, img = Bionoi(args.mol, args.cmap, args.out, args.dpi, args.alpha)
    #atoms, vor, img = cProfile.run('Bionoi(args.mol, args.cmap, args.out, args.dpi, args.alpha)')


