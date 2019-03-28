from scipy.spatial import cKDTree
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


def unpack_dict(data_dict):
    atoms = data_dict['atoms']
    vor = data_dict['vor']
    saliencyMap = data_dict['saliencyMap']  # the heatmap
    trueClass = data_dict['trueClass']  # the real class of this mol
    predClass = data_dict['predClass']  # the predicted class of this mol
    return atoms, vor, saliencyMap, trueClass, predClass


def unpack_pkl(pklFile):
    with open(pklFile, 'rb') as f:
        data_dict = load(f)
    dict_xoy_p = data_dict['xoy_+']
    dict_xoy_n = data_dict['xoy_-']
    dict_yoz_p = data_dict['yoz_+']
    dict_yoz_n = data_dict['yoz_-']
    dict_zox_p = data_dict['zox_+']
    dict_zox_n = data_dict['zox_-']
    info_xoy_p = unpack_dict(dict_xoy_p)
    info_xoy_n = unpack_dict(dict_xoy_n)
    info_yoz_p = unpack_dict(dict_yoz_p)
    info_yoz_n = unpack_dict(dict_yoz_n)
    info_zox_p = unpack_dict(dict_zox_p)
    info_zox_n = unpack_dict(dict_zox_n)
    return info_xoy_p, info_xoy_n, info_yoz_p, info_yoz_n, info_zox_p, info_zox_n


if __name__ == "__main__":
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-pickle',   
                        required = True, 
                        help='pickle object from Bionoi')
    parser.add_argument('-out',   
                        required = True,
                        help='path to output pickle (ends in \'.pkl\')')
    
    args = parser.parse_args()

    with open(args.pickle, 'rb') as f:
        data_dict = load(f)

    # Loop through atoms dataframes, sum their cells, then add Cell Sum column to merged_df
    merged_df = data_dict["xoy_+"]["atoms"].drop(labels=["P(x)", "P(y)", "polygons", "color"], axis=1)
    for key, val in data_dict.items():
        # Get variables
        atoms = val['atoms']
        vor = val['vor']
        heatmap = val['saliencyMap']

        # Dropping prevents build up of column space
        atoms = atoms.drop(labels=["P(x)", "P(y)", "polygons", "color"], axis=1)

        # Sum and store
        s = sumCells(vor, heatmap)
        tmp_df = merge(s, atoms)

        # Warning : this merging assumes that the different DataFrames are ordered exactly the same
        # TODO : Merge by column instead of adding new column
        merged_df["Cell Sum %s" % key] = tmp_df["Cell Sum"]

    # Sum the cells by atom
    final = merged_df.copy()
    final["total"] = merged_df[[col for col in merged_df.columns if col.startswith("Cell Sum")]].sum(axis=1)
    final = final.drop([col for col in final.columns if col.startswith("Cell Sum")], axis=1)
    final = final.sort_values("total", ascending=False)

    with open(args.out, "wb") as pkl:
        dump(final, pkl)
