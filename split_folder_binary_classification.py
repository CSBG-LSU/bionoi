'''
    Split the target folder to perform binary classification
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
    parser.add_argument('-opMode',
                        default='control_vs_nucleotide',
                        required=False,
                        help='control_vs_heme or control_vs_nucleotide')
    parser.add_argument('-sourceFolder',
                        default='../mols_extract/',
                        required=False,
                        help='source .mol2 folder path')
    parser.add_argument('-targetFolder',
                        default='../mols_extract_binary_classification/',
                        required=False,
                        help='target .mol2 folder path for cross-validation')
    parser.add_argument('-valFration',
                        type=float,
                        default=0.1,
                        required=False,
                        help='validation data set ratio')
    parser.add_argument('-testFraction',
                        type=float,
                        default=0.1,
                        required=False,
                        help='test data set ratio')
    return parser.parse_args()

# save the splitted datasets to folders
def saveToFolder(fileCollection, sourceFolder, targetFolder):
    m = len(fileCollection)
    for i in range(m):
        source = sourceFolder + '/' + fileCollection[i]
        target = targetFolder + '/' + fileCollection[i]
        copyfile(source, target)

def split2BC_control_vs_nucleotide(sourceFolder, targetFolder, valFraction=0.1, testFraction=0.1):
    sourceFolderControl = sourceFolder + 'control/'
    sourceFolderNucleotide = sourceFolder + 'nucleotide/'

    # a list of files in control
    fileListControl = [f for f in listdir(sourceFolderControl) if isfile(join(sourceFolderControl, f))]
    numControl = len(fileListControl)
    # shuffle the collection to create a new one
    idxControl = np.arange(numControl)
    idxPermuControl = np.random.permutation(idxControl)
    colShuffleControl = [fileListControl[i] for i in idxPermuControl]

    # a list of files in nucleotide
    fileListNucleotide = [f for f in listdir(sourceFolderNucleotide) if isfile(join(sourceFolderNucleotide, f))]
    numNucleotide = len(fileListNucleotide)
    # shuffle the collection to create a new one
    idxNucleotide = np.arange(numNucleotide)
    idxPermuNucleotide = np.random.permutation(idxNucleotide)
    colShuffleNucleotide = [fileListNucleotide[i] for i in idxPermuNucleotide]

    # calculate the length of each new folder of control
    lenValControl = int(numControl*valFration)
    print('lenth of val data of control:', lenValControl)
    lenTestControl = int(numControl*testFraction)
    print('lenth of test data of control:', lenTestControl)
    lenTrainControl = numControl - (lenValControl + lenTestControl)
    print('length of train data of control:',lenTrainControl)

    # calculate the length of each new folder of nucleotide
    lenValNucleotide = int(numNucleotide*valFration)
    print('lenth of val data of nucleotide:', lenValNucleotide)
    lenTestNucleotide = int(numNucleotide*testFraction)
    print('lenth of test data of nucleotide:', lenTestNucleotide)
    lenTrainNucleotide = numNucleotide - (lenValNucleotide + lenTestNucleotide)
    print('length of train data of nucleotide:',lenTrainNucleotide)

    # create 3 new lists of images of control
    trainSetControl = colShuffleControl[0:lenTrainControl]
    valSetControl = colShuffleControl[lenTrainControl:lenTrainControl + lenValControl]
    testSetControl = colShuffleControl[lenTrainControl + lenValControl:numControl]

    # create 3 new lists of images of nucleotide
    trainSetNucleotide = colShuffleNucleotide[0:lenTrainNucleotide]
    valSetNucleotide = colShuffleNucleotide[lenTrainNucleotide:lenTrainNucleotide + lenValNucleotide]
    testSetNucleotide = colShuffleNucleotide[lenTrainNucleotide + lenValNucleotide:numNucleotide]

    # create sub-directory in targetFolder of control
    trainDirControl = targetFolder+'train/'+'control/'
    if not os.path.exists(trainDirControl):
        os.makedirs(trainDirControl)
    valDirControl = targetFolder+'val/'+'control/'
    if not os.path.exists(valDirControl):
        os.makedirs(valDirControl)
    testDirControl = targetFolder+'test/'+'control/'
    if not os.path.exists(testDirControl):
        os.makedirs(testDirControl)

    # create sub-directory in targetFolder of nucleotide
    trainDirNucleotide = targetFolder+'train/'+'nucleotide/'
    if not os.path.exists(trainDirNucleotide):
        os.makedirs(trainDirNucleotide)
    valDirNucleotide = targetFolder+'val/'+'nucleotide/'
    if not os.path.exists(valDirNucleotide):
        os.makedirs(valDirNucleotide)
    testDirNucleotide = targetFolder+'test/'+'nucleotide/'
    if not os.path.exists(testDirNucleotide):
        os.makedirs(testDirNucleotide)

    # copy files to corresponding new folder
    saveToFolder(trainSetControl, sourceFolderControl, trainDirControl)
    saveToFolder(valSetControl, sourceFolderControl, valDirControl)
    saveToFolder(testSetControl, sourceFolderControl, testDirControl)
    saveToFolder(trainSetNucleotide, sourceFolderNucleotide, trainDirNucleotide)
    saveToFolder(valSetNucleotide, sourceFolderNucleotide, valDirNucleotide)
    saveToFolder(testSetNucleotide, sourceFolderNucleotide, testDirNucleotide)

def split2BC_control_vs_heme(sourceFolder, targetFolder, valFration=0.1, testFraction=0.1):
    sourceFolderControl = sourceFolder + 'control/'
    sourceFolderHeme = sourceFolder + 'heme/'

    # a list of files in control
    fileListControl = [f for f in listdir(sourceFolderControl) if isfile(join(sourceFolderControl, f))]
    numControl = len(fileListControl)
    # shuffle the collection to create a new one
    idxControl = np.arange(numControl)
    idxPermuControl = np.random.permutation(idxControl)
    colShuffleControl = [fileListControl[i] for i in idxPermuControl]

    # a list of files in heme
    fileListHeme = [f for f in listdir(sourceFolderHeme) if isfile(join(sourceFolderHeme, f))]
    numHeme = len(fileListHeme)
    # shuffle the collection to create a new one
    idxHeme = np.arange(numHeme)
    idxPermuHeme = np.random.permutation(idxHeme)
    colShuffleHeme = [fileListHeme[i] for i in idxPermuHeme]

    # calculate the length of each new folder of control
    lenValControl = int(numControl*valFration)
    print('lenth of val data of control:', lenValControl)
    lenTestControl = int(numControl*testFraction)
    print('lenth of test data of control:', lenTestControl)
    lenTrainControl = numControl - (lenValControl + lenTestControl)
    print('length of train data of control:',lenTrainControl)

    # calculate the length of each new folder of heme
    lenValHeme = int(numHeme*valFration)
    print('lenth of val data of heme:', lenValHeme)
    lenTestHeme = int(numHeme*testFraction)
    print('lenth of test data of heme:', lenTestHeme)
    lenTrainHeme = numHeme - (lenValHeme + lenTestHeme)
    print('length of train data of heme:',lenTrainHeme)

    # create 3 new lists of images of control
    trainSetControl = colShuffleControl[0:lenTrainControl]
    valSetControl = colShuffleControl[lenTrainControl:lenTrainControl + lenValControl]
    testSetControl = colShuffleControl[lenTrainControl + lenValControl:numControl]

    # create 3 new lists of images of heme
    trainSetHeme = colShuffleHeme[0:lenTrainHeme]
    valSetHeme = colShuffleHeme[lenTrainHeme:lenTrainHeme + lenValHeme]
    testSetHeme = colShuffleHeme[lenTrainHeme + lenValHeme:numHeme]

    # create sub-directory in targetFolder of control
    trainDirControl = targetFolder+'train/'+'control/'
    if not os.path.exists(trainDirControl):
        os.makedirs(trainDirControl)
    valDirControl = targetFolder+'val/'+'control/'
    if not os.path.exists(valDirControl):
        os.makedirs(valDirControl)
    testDirControl = targetFolder+'test/'+'control/'
    if not os.path.exists(testDirControl):
        os.makedirs(testDirControl)

    # create sub-directory in targetFolder of heme
    trainDirHeme = targetFolder+'train/'+'heme/'
    if not os.path.exists(trainDirHeme):
        os.makedirs(trainDirHeme)
    valDirHeme = targetFolder+'val/'+'heme/'
    if not os.path.exists(valDirHeme):
        os.makedirs(valDirHeme)
    testDirHeme = targetFolder+'test/'+'heme/'
    if not os.path.exists(testDirHeme):
        os.makedirs(testDirHeme)

    # copy files to corresponding new folder
    saveToFolder(trainSetControl, sourceFolderControl, trainDirControl)
    saveToFolder(valSetControl, sourceFolderControl, valDirControl)
    saveToFolder(testSetControl, sourceFolderControl, testDirControl)
    saveToFolder(trainSetHeme, sourceFolderHeme, trainDirHeme)
    saveToFolder(valSetHeme, sourceFolderHeme, valDirHeme)
    saveToFolder(testSetHeme, sourceFolderHeme, testDirHeme)

if __name__ == "__main__":
    args = getArgs()
    opMode = args.opMode
    sourceFolder = args.sourceFolder
    targetFolder = args.targetFolder
    valFration = args.valFration
    testFraction = args.testFraction
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)
    if opMode == 'control_vs_heme':
        split2BC_control_vs_heme(sourceFolder, targetFolder, valFration, testFraction)
    elif opMode == 'control_vs_nucleotide':
        split2BC_control_vs_nucleotide(sourceFolder, targetFolder, valFration, testFraction)
    else:
        print('error: input opMode does NOT exist.')
