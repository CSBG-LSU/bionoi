from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import pandas as pd
import matplotlib
import sys,os,argparse
from biopandas.mol2 import PandasMol2
import matplotlib.pyplot as plt

# Adapted from https://gist.github.com/pv/8036995

def voronoi_polygons_2D(vor, radius=None):
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
        if all([v >= 0 for v in vertices]):
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

def voronoi_atoms(bs,bs_out=None,size=None):
    pd.options.mode.chained_assignment = None
    
    # read molecules in mol2 format 
    atoms = PandasMol2().read_mol2(bs)
    pt = atoms.df[['subst_name','atom_type', 'atom_name','x','y','z']]
    
    # convert 3D to 2D 
    pt.loc[:,'X'] = pt.x/abs(pt.z)**.5
    pt.loc[:,'Y'] = pt.y/abs(pt.z)**.5
    
    # setting output image size, labels off, set 120 dpi w x h
    size = 120 if size is None else size
    figure = plt.figure(figsize=(2.69 , 2.70),dpi=int(size))
    ax = plt.subplot(111); ax.axis('off') ;ax.tick_params(axis='both', bottom='off', left='off',right='off',labelleft='off', labeltop='off',labelright='off', labelbottom='off')

    # compute Voronoi tesselation
    vor = Voronoi(pt[['X','Y']])
    regions, vertices = voronoi_polygons_2D(vor)
    polygons = []
    for i in regions:
        polygon = vertices[i]
        polygons.append(polygon)
    pt.loc[:,'polygons'] = polygons
    
    # color by atom types
    atom_color = {'C.3':'#006600','N.3':'#000066','O.3':'#660000','S.3':'#666600','C.ar':'#009900','N.ar':'#000099','C.2':'#00CC00','N.2':'#0000CC','O.2':'#C00000','C.cat': '#00FF00','N.am':'#3333FF','N.2':'#3333FF','N.pl3':'#6666FF','O.co2':'#FF9999'}         
    for i, row in pt.iterrows():
        tmp1 = pt.loc[i][['atom_type']][0]
        tmp3 = np.array(pt.loc[i][['polygons']])[0]
        col1 = atom_color[tmp1]
        p1 = matplotlib.patches.Polygon(tmp3, facecolor=col1, edgecolor='black', alpha=0.5)
        ax.add_patch(p1)
    ax.set_xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
    ax.set_ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
    
    # output image saving in any format; default jpg
    bs_out = 'out.jpg' if bs_out is None else bs_out
    plt.savefig(bs_out, frameon=False,bbox_inches="tight", pad_inches=False)
    return None

def myargs():
    parser = argparse.ArgumentParser('python')                                              
    parser.add_argument('-mol', required = True, help = 
                        'location of the protein/ligand mol2 file path')
    parser.add_argument('-out', required = False, help = 'location for the image to be saved')
    parser.add_argument('-dpi', required = False, help = 'image quality in dpi, eg: 300')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = myargs()
    voronoi_atoms(args.mol,args.out,args.dpi)
