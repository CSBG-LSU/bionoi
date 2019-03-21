from scipy.spatial import cKDTree
from scipy.spatial import Voronoi
from collections import defaultdict 
import numpy as np
import argparse
from pickle import load, dump
import pandas as pd

def sumCells(vor, heatmap):
    # Generate all points to loop through 
    x = np.linspace(vor.min_bound[0], vor.max_bound[0], heatmap.shape[0])
    y = np.linspace(vor.min_bound[1], vor.max_bound[1], heatmap.shape[1])
    points = np.array([(xi, yi) for xi in x for yi in y])
    
    # Make k-d tree for point lookup 
    voronoi_kdtree = cKDTree(vor.points)
    dist, point_regions = voronoi_kdtree.query(points, k=1)
    
    # Sum scores in heatmap for each voronoi cell 
    cells = defaultdict(float)
    for pointi, reg in enumerate(point_regions):
        # Convert point index to i,j pairs from the image 
        # TODO: The '%' conversion might not be perfect arithmetic 
        i, j = pointi//heatmap.shape[0], pointi % heatmap.shape[1]
        
        cells[list(vor.point_region).index(vor.point_region[reg])] += heatmap[i][j]
        
    return cells


def merge(cellsum, atomsdf):
    sums = pd.DataFrame.from_dict(cellsum, orient='index', columns=["Cell Sum"])
    return pd.merge(atomsdf, sums, left_index=True, right_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-pickle',   
                        required = True, 
                        help='pickle object from Bionoi')
    parser.add_argument('-heatmap',   
                        required=True,
                        help='the heatmap as pickled numpy array ')
    parser.add_argument('-out',   
                        required = True, 
                        help='path to output pickle (ends in \'.pkl\')')
    
    args = parser.parse_args()
    
    with open(args.pickle, "rb") as pkl:
        atoms, voronoi = load(pkl)
    with open(args.heatmap, "rb") as pkl:
        heatmap = load(pkl)
    
    s = sumCells(voronoi, heatmap)
    out = merge(s, atoms)
    
    with open(args.out, "wb") as pkl:
        dump(out, pkl)
    
    
    
