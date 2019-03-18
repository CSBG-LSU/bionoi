from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import pandas as pd
import matplotlib

from biopandas.mol2 import PandasMol2
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sklearn.cluster import KMeans
from math import sqrt, asin, atan2, log, pi, tan

from alignment import *


def k_different_colors(k: int):
    colors = dict(**mcolors.CSS4_COLORS)

    rgb = lambda color: mcolors.to_rgba(color)[:3]
    hsv = lambda color: mcolors.rgb_to_hsv(color)

    col_dict = [(k, rgb(k)) for c, k in colors.items()]
    X = np.array([j for i, j in col_dict])

    # Perform kmeans on rqb vectors
    kmeans = KMeans(n_clusters=k)
    kmeans = kmeans.fit(X)
    # Getting the cluster labels
    labels = kmeans.predict(X)
    # Centroid values
    C = kmeans.cluster_centers_

    # Find one color near each of the k cluster centers
    closest_colors = np.array([np.sum((X - C[i]) ** 2, axis=1) for i in range(C.shape[0])])
    keys = sorted(closest_colors.argmin(axis=1))

    return [col_dict[i][0] for i in keys]


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
        radius = vor.points.ptp().max() * 2

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

            t = vor.points[p2] - vor.points[p1]  # tangent
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
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
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


def miller(x, y, z):
    radius = sqrt(x ** 2 + y ** 2 + z ** 2)
    latitude = asin(z / radius)
    longitude = atan2(y, x)
    lat = 5 / 4 * log(tan(pi / 4 + 2 / 5 * latitude))
    return lat, longitude

"""
return transformation coordinates(matrix: X*3) 
Principal Axes Alignment
"""
def alignment(pocket, proDirct):

    pocket_coords = np.array([pocket.x, pocket.y, pocket.z]).T
    pocket_center = np.mean(pocket_coords, axis=0)  # calculate mean of each column
    pocket_coords = pocket_coords - pocket_center   # Centralization
    inertia = np.cov(pocket_coords.T)               # get covariance matrix (of centralized data)
    e_values, e_vectors = np.linalg.eig(inertia)    # linear algebra eigenvalue eigenvector
    sorted_index = np.argsort(e_values)[::-1]       # sort eigenvalues (increase)and reverse (decrease)
    sorted_vectors = e_vectors[:, sorted_index]

    if proDirct == 1:
        transformation_matrix = xoy_positive_proj(sorted_vectors)
    elif proDirct == 2:
        transformation_matrix = xoy_negative_proj(sorted_vectors)
    elif proDirct == 3:
        transformation_matrix = yoz_positive_proj(sorted_vectors)
    elif proDirct == 4:
        transformation_matrix = yoz_negative_proj(sorted_vectors)
    elif proDirct == 5:
        transformation_matrix = zox_positive_proj(sorted_vectors)
    elif proDirct == 6:
        transformation_matrix = zox_negative_proj(sorted_vectors)

    transformed_coords = (np.matmul(transformation_matrix, pocket_coords.T)).T
    # transformed_coords = (transformation_matrix.dot(pocket_coords.T)).T

    return transformed_coords


def voronoi_atoms(bs, cmap, colorby, bs_out=None, size=None, dpi=None, alpha=1, save_fig=True,
                    projection=miller, proDirct=None):
    # Suppresses warning
    pd.options.mode.chained_assignment = None

    # Read molecules in mol2 format
    mol2 = PandasMol2().read_mol2(bs)
    atoms = mol2.df[['subst_id', 'subst_name', 'atom_type', 'atom_name', 'x', 'y', 'z']]
    atoms.columns = ['res_id', 'residue_type', 'atom_type', 'atom_name', 'x', 'y', 'z']
    atoms['residue_type'] = atoms['residue_type'].apply(lambda x: x[0:3])

    # Align to principal Axis
    trans_coords = alignment(atoms, proDirct)  # get the transformation coordinate
    atoms['x'] = trans_coords[:, 0]
    atoms['y'] = trans_coords[:, 1]
    atoms['z'] = trans_coords[:, 2]

    # convert 3D  to 2D
    atoms["P(x)"] = atoms[['x', 'y', 'z']].apply(lambda coord: projection(coord.x, coord.y, coord.z)[0], axis=1)
    atoms["P(y)"] = atoms[['x', 'y', 'z']].apply(lambda coord: projection(coord.x, coord.y, coord.z)[1], axis=1)

    # setting output image size, labels off, set 120 dpi w x h
    size = 128 if size is None else size
    dpi = 120 if dpi is None else dpi

    figure = plt.figure(figsize=(int(size) / int(dpi), int(size) / int(dpi)), dpi=int(dpi))
    # figsize is in inches, dpi is the resolution of the figure
    ax = plt.subplot(111)
    # default is (111)

    ax.axis('off')
    ax.tick_params(axis='both', bottom=False, left=False, right=False,
                   labelleft=False, labeltop=False,
                   labelright=False, labelbottom=False)

    # Compute Voronoi tesselation
    vor = Voronoi(atoms[['P(x)', 'P(y)']])
    regions, vertices = voronoi_finite_polygons_2d(vor)
    polygons = []
    for reg in regions:
        polygon = vertices[reg]
        polygons.append(polygon)
    atoms.loc[:, 'polygons'] = polygons

    # Check alpha
    alpha = float(alpha)

    # Color by colorby
    if colorby in ["atom_type", "residue_type"]:
        colors = [cmap[_type]["color"] for _type in atoms[colorby]]
    elif colorby == "residue_num":
        cmap = k_different_colors(len(set(atoms["res_id"])))
        cmap = {res_num: color for res_num, color in zip(set(atoms["res_id"]), cmap)}
        colors = atoms["res_id"].apply(lambda x: cmap[x])
    else:
        raise ValueError
    atoms["color"] = colors

    for i, row in atoms.iterrows():
        colored_cell = matplotlib.patches.Polygon(row["polygons"],
                                                  facecolor=row['color'],
                                                  edgecolor=row['color'],
                                                  alpha=alpha,
                                                  linewidth=0.2)
        ax.add_patch(colored_cell)

    # atoms.loc[:,"color"] = colors

    ax.set_xlim(vor.min_bound[0], vor.max_bound[0])
    ax.set_ylim(vor.min_bound[1], vor.max_bound[1])

    # Output image saving in any format; default jpg
    bs_out = 'out.jpg' if bs_out is None else bs_out

    # Get image as numpy array
    figure.tight_layout(pad=0)
    img = fig_to_numpy(figure, alpha=alpha)

    if save_fig:
        plt.subplots_adjust(bottom=0, top=1, left=0, right=1)
        plt.savefig(bs_out, frameon=False, pad_inches=False)

    return atoms, vor, img


"""
atoms, vor, img = Bionoi(args.mol, args.out, args.size, args.dpi, args.alpha, args.colorby,
                         args.proDirect,
                         args.rotAngle2D,
                         args.flip,
                         args.trainPercent,
                         args.validatePercent,
                         args.testPercent)
"""


def Bionoi(mol, bs_out, size, dpi, alpha, colorby, proDirct):
    if colorby in ["atom_type", "residue_type"]:
        cmap = "./cmaps/atom_cmap.csv" if colorby == "atom_type" else "./cmaps/res_hydro_cmap.csv"

        # Check for color mapping file, make dict
        try:
            with open(cmap, "rt") as cMapF:
                # Parse color map file
                cmap = np.array(
                    [line.replace("\n", "").split(";") for line in cMapF.readlines() if not line.startswith("#")])
                # To dict
                cmap = {code: {"color": color, "definition": definition} for code, definition, color in cmap}
        except FileNotFoundError:
            raise FileNotFoundError(
                "Color mapping file not found. Be sure to specify YOURPATH/cmaps/ before the cmap basename.")
        except ValueError:
            raise ValueError(
                "Error while parsing cmap file. Check the file's delimeters and compare to examples in cmaps folder")
    else:
        cmap = None

    # Run
    atoms, vor, img = voronoi_atoms(mol, cmap, colorby,
                                    bs_out=bs_out,
                                    size=size, dpi=dpi,
                                    alpha=alpha,
                                    save_fig=False,
                                    proDirct=proDirct)

    return atoms, vor, img
