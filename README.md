# bionoi
Conversion of biomolecules to Voronoi diagrams.

## Description
bionoi constructs the [voronoi diagram](https://en.wikipedia.org/wiki/Voronoi_diagram) (VD) of protein binding site/pocket or ligand structure based on the 3D or 2D coordinate structure.

## Requirements
1. Python 3
2. numpy 1.14+
3. scipy 0.18+
4. Pandas 0.19+
5. scikit-spatial 0.12.0
6. matplotlib 2.0.2+
7. biopandas ; eg : pip/conda install biopandas

## Getting started

1. copy/download the code from GitHub
2. If input is a 3D coordinate of protein/ligand, it will be projected to 2D plane
3. Run voronoi.py with a .mol2 file (see Examples)

## Examples

create a 2D image using mol2 file in 2D or 3D format

    ./voronoi.py
    or
    ./voronoi.py -mol a.mol2 -out a.jpg -dpi 120 -alpha 0.5

An example of voronoi image of ATP-binding site protein pocket colored by atom types:

![eg_image](https://github.com/rajiv03/DeepDrugV/blob/master/Voronoi_2D_4v94E.jpg)

chaperonin (4v94, chain E)

Contributors:

Rajiv Gandhi Govindaraj, Jeffrey Lemoine, Limeng Pu, Ye Fang and Michal Brylinski.
