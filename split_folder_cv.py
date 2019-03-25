'''
    Split the target folder to perform k-fold cross-validation
'''
import numpy as np
import os
import sys
import argparse
from os import listdir
from os.path import isfile, join
from shutil import copyfile

def getArgs():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-k',
                        type=int,
                        default=10,
                        required=False,
                        help='number of folds')
    parser.add_argument('-opMode',
                        default='control_vs_heme',
                        required=False,
                        help='operation mode, control_vs_heme or control_vs_nucleotide')
    parser.add_argument('-sourceFolder',
                        default='../mols_extract/',
                        required=False,
                        help='source .mol2 folder path')
    parser.add_argument('-targetFolder',
                        default='../mols_extract_cv/',
                        required=False,
                        help='target .mol2 folder path for cross-validation')
    return parser.parse_args()

# save the splitted datasets to folders
def saveToFolder(fileCollection, sourceFolder, targetFolder):
    m = len(fileCollection)
    for i in range(m):
        source = sourceFolder + '/' + fileCollection[i]
        target = targetFolder + '/' + fileCollection[i]
        copyfile(source, target)

def split2CVFolder(k, sourceFolder, targetFolder, type):
    """
    type: 'control/', 'heme/', 'nucleotide/' or 'steroid/'
    """
    sourceFolder = sourceFolder + type

    # a list of files in target folder
    fileList = [f for f in listdir(sourceFolder) if isfile(join(sourceFolder, f))]
    m = len(fileList)

    # shuffle the collection to create a new one
    idx = np.arange(m)
    idx_permu = np.random.permutation(idx)
    colShuffle = [fileList[i] for i in idx_permu]

    # calculate the length of each new folder
    lenVal = int(m/k)
    print('lenth of val data:', lenVal)
    lenTrain = m - lenVal
    print('length of train data:',lenTrain)

    # split the collection in a k-fold cross validation manner
    for i in range (k):
        print('i:',i)
        colVal = colShuffle[i*lenVal:(i+1)*lenVal]
        print('val idx:', i*lenVal,' to ', (i+1)*lenVal-1)
        colTrain = colShuffle[0:i*lenVal] + colShuffle[(i+1)*lenVal:m]
        print('length of train:',len(colTrain))
        print('length of val:',len(colVal))

        # create sub-directory in targetFolder for current fold
        trainDir = targetFolder+'/cv'+str(i+1)+'/train/'+type
        if not os.path.exists(trainDir):
            os.makedirs(trainDir)
        valDir = targetFolder+'/cv'+str(i+1)+'/val/'+type
        if not os.path.exists(valDir):
            os.makedirs(valDir)

        # copy files to corresponding new folder
        saveToFolder(colTrain, sourceFolder, trainDir)
        saveToFolder(colVal, sourceFolder, valDir)
        print('-------------------------------------')

if __name__ == "__main__":
    args = getArgs()
    k = args.k
    opMode = args.opMode
    sourceFolder = args.sourceFolder
    targetFolder = args.targetFolder
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)
    if opMode == 'control_vs_heme':
        split2CVFolder(k,sourceFolder,targetFolder,type = '/control')
        split2CVFolder(k,sourceFolder,targetFolder,type = '/heme')
    elif opMode == 'control_vs_nucleotide':
        split2CVFolder(k,sourceFolder,targetFolder,type = '/control')
        split2CVFolder(k,sourceFolder,targetFolder,type = '/nucleotide')
    else:
        print('error: invalid opMode')

    #split2CVFolder(k,sourceFolder,targetFolder,type = '/control')
    #split2CVFolder(k,sourceFolder,targetFolder,type = '/heme')
    #split2CVFolder(k,sourceFolder,targetFolder,type = '/nucleotide')
    #split2CVFolder(k,sourceFolder,targetFolder,type = '/steroid')
