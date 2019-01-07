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

# Principal Axes Alignment
def normalize(v):
    """ vector normalization """
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def vrrotvec(a,b):
    """ Function to rotate one vector to another, inspired by
    vrrotvec.m in MATLAB """
    a = normalize(a)
    b = normalize(b)
    ax = normalize(np.cross(a,b))
    angle = np.arccos(np.minimum(np.dot(a,b),[1]))
    if not np.any(ax):
        absa = np.abs(a)
        mind = np.argmin(absa)
        c = np.zeros((1,3))
        c[mind] = 0
        ax = normalize(np.cross(a,c))
    r = np.concatenate((ax,angle))
    return r

def vrrotvec2mat(r):
    """ Convert the axis-angle representation to the matrix representation of the 
    rotation """
    s = np.sin(r[3])
    c = np.cos(r[3])
    t = 1 - c

    n = normalize(r[0:3])

    x = n[0]
    y = n[1]
    z = n[2]

    m = np.array(
     [[t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
     [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
     [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]]
    )
    return m

def alignment(pocket):
    pocket_coords = np.array([pocket.x, pocket.y, pocket.z]).T
    pocket_center = np.mean(pocket_coords, axis = 0)
    pocket_coords = pocket_coords - pocket_center
    inertia = np.cov(pocket_coords.T)
    e_values, e_vectors = np.linalg.eig(inertia)
    sorted_index = np.argsort(e_values)[::-1]
    sorted_vectors = e_vectors[:,sorted_index]
    # Align the first principal axes to the X-axes
    rx = vrrotvec(np.array([1,0,0]),sorted_vectors[:,0])
    mx = vrrotvec2mat(rx)
    pa1 = np.matmul(mx.T,sorted_vectors)
    # Align the second principal axes to the Y-axes
    ry = vrrotvec(np.array([0,1,0]),pa1[:,1])
    my = vrrotvec2mat(ry)
    transformation_matrix = np.matmul(my.T,mx.T)
    # transform the protein coordinates to the center of the pocket and align with the principal
    # axes with the pocket
    transformed_coords = (np.matmul(transformation_matrix,pocket_coords.T)).T
    return transformed_coords

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

from math import sqrt, asin, atan2,log, pi, tan
def miller(x,y,z):
    radius = sqrt( x**2 + y**2 + z**2 )
    latitude = asin( z / radius )
    longitude = atan2( y, x)
    lat = 5/4 * log(tan(pi/4 + 2/5 * latitude))
    return lat, longitude

def voronoi_atoms(bs, cmap, colorby,bs_out=None, size=None, dpi=None, alpha=0.5, save_fig=True, projection=miller):
    # Suppresses warning
    pd.options.mode.chained_assignment = None

    # Read molecules in mol2 format
    mol2 = PandasMol2().read_mol2(bs)
    atoms = mol2.df[['subst_id','subst_name','atom_type', 'atom_name','x','y','z']]
    atoms.columns = ['res_id','residue_type','atom_type', 'atom_name','x','y','z']
    atoms['residue_type'] = atoms['residue_type'].apply(lambda x: x[0:3])

    # Align to principal axis
    trans_coords = alignment(atoms)
    atoms['x'] = trans_coords[:,0]
    atoms['y'] = trans_coords[:,1]
    atoms['z'] = trans_coords[:,2]
    
    # convert 3D  to 2D
    atoms["P(x)"] = atoms[['x','y','z']].apply(lambda coord: projection(coord.x,coord.y,coord.z)[0], axis=1)
    atoms["P(y)"] = atoms[['x','y','z']].apply(lambda coord: projection(coord.x,coord.y,coord.z)[1], axis=1)

    # setting output image size, labels off, set 120 dpi w x h
    size = 128 if size is None else size
    dpi = 120 if dpi is None else dpi

    figure = plt.figure(figsize=(int(size)/int(dpi) , int(size)/int(dpi)), dpi=int(dpi))

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

    # Color by colorby 
    if colorby in ["atom_type","residue_type"]:
        colors = [cmap[_type]["color"] for _type in atoms[colorby]]
    elif colorby=="residue_num":
        cmap = k_different_colors(len(set(atoms["res_id"])))
        cmap = {res_num:color for res_num,color in zip(set(atoms["res_id"]),cmap)}
        colors = atoms["res_id"].apply(lambda x: cmap[x])       
    else:
        raise ValueError
    atoms["color"] = colors
    
    for i, row in atoms.iterrows():
        colored_cell = matplotlib.patches.Polygon(row["polygons"],
                                        facecolor = row['color'],
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
        plt.subplots_adjust(bottom=0, top=1, left=0, right=1)
        plt.savefig(bs_out, frameon=False, pad_inches=False)

    return atoms, vor, img

def getArgs():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-mol',   
                        default="./examples/4v94E.mol2",      
                        required = False, 
                        help = 'the protein/ligand mol2 file')
    parser.add_argument('-out',   
                        default="./out.jpg",         
                        required = False, 
                        help = 'the output image file')
    parser.add_argument('-dpi',   
                        default="50",               
                        required = False, 
                        help = 'image quality in dpi')
    parser.add_argument('-size',  default="128",               
                        required = False, 
                        help = 'image size in pixels, eg: 128')
    parser.add_argument('-alpha', 
                        default="0.5",               
                        required = False, 
                        help = 'alpha for color of cells')
    parser.add_argument('-colorby', 
                        default="atom_type",  
                        choices=["atom_type","residue_type","residue_num"],
                        required = False, 
                        help = 'color the voronoi cells according to {atom_type, residue_type, residue_num}')

    return parser.parse_args()

def Bionoi(mol, bs_out, size, dpi, alpha, colorby):
    if colorby in ["atom_type","residue_type"]:
        cmap = "./cmaps/atom_cmap.csv" if colorby=="atom_type" else "./cmaps/res_cmap.csv"
        
        # Check for color mapping file, make dict
        try:
            with open(cmap,"rt") as cMapF:
                # Parse color map file
                cmap  = np.array([line.replace("\n","").split(";") for line in cMapF.readlines() if not line.startswith("#")])
                # To dict
                cmap = {code:{"color":color, "definition":definition} for code, definition, color in cmap}
        except FileNotFoundError:
            raise FileNotFoundError("Color mapping file not found. Be sure to specify YOURPATH/cmaps/ before the cmap basename.")
        except ValueError:
            raise ValueError("Error while parsing cmap file. Check the file's delimeters and compare to examples in cmaps folder")
    else:
        cmap = None
    
    # Run
    atoms, vor, img = voronoi_atoms(mol, cmap, colorby,bs_out=bs_out, size=size, dpi=dpi, alpha=alpha, save_fig=True)

    return atoms, vor, img



if __name__ == "__main__":
    args = getArgs()
    atoms, vor, img = Bionoi(args.mol, args.out, args.size, args.dpi, args.alpha, args.colorby)
    #atoms, vor, img = cProfile.run('Bionoi(args.mol, args.cmap, args.out, args.dpi, args.alpha)')
    
