"""
Copy the scores of atoms and paste into the last column of a .pdb file of a pocket.
"""
import os
import argparse
import csv
import pandas as pd
from shutil import copyfile

def pocket_visual_single(pdb_dir, source_pdb, scores_dir, scores, out_dir):
    """
    take the source pdb file and its corresponding score, then add the scores 
    to the last column of the pdb file, save it to out_dir.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)    
    pdb_file = pdb_dir + source_pdb
    scores_file = scores_dir + scores
    out_file = out_dir + source_pdb

    # copy file to out_dir
    copyfile(pdb_file, out_file)

    # read the scores as a list
    df = pd.read_csv(scores_file)
    scores = df['total']
    #print(scores)

    # modify the pdb file

if __name__ == "__main__":
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-op',   
                        required = False,
                        default = 'control_vs_heme',
                        choices = ['control_vs_heme', 'control_vs_nucleotide'],
                        help='operation modes')    
    parser.add_argument('-final_scores_dir_heme',   
                        required = False,
                        default = '../../analyse/final_scores/control_vs_heme/test/heme/',
                        help='final scores directory for heme')
    parser.add_argument('-final_scores_dir_nucleotide',   
                        required = False,
                        default = '../../analyse/final_scores/control_vs_nucleotide/test/nucleotide/',
                        help='final scores directory for nucleotide')
    parser.add_argument('-pdb_dir_heme',   
                        required = False,
                        default = '../../analyse/pocket-lpc-heme/',
                        help='heme pdb directory')
    parser.add_argument('-pdb_dir_nucleotide',   
                        required = False,
                        default = '../../analyse/pocket-lpc-nucleotide/',
                        help='nucleotide pdb directory')                    
    parser.add_argument('-out_dir_heme',   
                        required = False,
                        default = '../../analyse/pdb_visual_heme/',
                        help='pdb output directory for heme')
    parser.add_argument('-out_dir_nucleotide',   
                        required = False,
                        default = '../../analyse/pdb_visual_nucleotide/',
                        help='pdb output directory for nucleotide')
    args = parser.parse_args()
    op = args.op
    final_scores_dir_heme = args.final_scores_dir_heme
    final_scores_dir_nucleotide = args.final_scores_dir_nucleotide
    pdb_dir_heme = args.pdb_dir_heme
    pdb_dir_nucleotide = args.pdb_dir_nucleotide
    out_dir_heme = args.out_dir_heme
    out_dir_nucleotide = args.out_dir_nucleotide
    if op == 'control_vs_heme':
        final_scores_dir = final_scores_dir_heme
        pdb_dir = pdb_dir_heme
        out_dir = out_dir_heme
    elif op == 'control_vs_nucleotide':
        final_scores_dir = final_scores_dir_nucleotide
        pdb_dir = pdb_dir_nucleotide
        out_dir = out_dir_nucleotide

    source_pdb = '1akkA00.pdb'
    scores = '1akkA00-1.csv'
    pocket_visual_single(pdb_dir, source_pdb, final_scores_dir, scores, out_dir)