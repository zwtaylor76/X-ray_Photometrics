import numpy as np
from scipy.interpolate import interp1d

datasetsFolder = 'D:/OneDrive/Documents/Stanford Stuff/Lab/Dresselhaus-Marais/Custom Python Scripts/Photometrics/Datasets/'

sourceList = {'CXRO': [0.2, 30], #keV, um
              'NIST':[2, 433], #keV, cm-1
              }

validMaterials = ['Ag', 'Al', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Ca', 'Cd', 'C-diamond', 
                  'C-glassy', 'Co', 'Cr', 'Cs', 'Cu', 'Fe', 'Ga', 'Ge', 'Hf', 'Hg',
                  'In', 'Ir', 'K', 'Li', 'Mg', 'Mn', 'Mo', 'Na', 'Nb', 'Ni',
                  'Os', 'Pb', 'Pd', 'Pt', 'Rb', 'Re', 'Rh', 'Ru', 'Sb', 'Sc',
                  'Si', 'Sn', 'Sr', 'Ta', 'Tc', 'Ti', 'Tl', 'U', 'V', 'W', 'Y',
                  'Zn', 'Zr', 'LuAG']

validCRLMaterials = ['Al', 'Be', 'C-diamond', 'Si', 'Ge']

sourceCRLList = {'Composite':[8,100]}

def getMat(mat, source):
    assert mat in validMaterials, "Material not in database."
    return np.loadtxt(f'{datasetsFolder}{source}_{mat}.txt')

def getCRLMat(mat, source):
    assert mat in validCRLMaterials, "Material not in database."
    return np.loadtxt(f'{datasetsFolder}Decriment_{source}_{mat}.txt')

def getDensities():
    return np.loadtxt(f'{datasetsFolder}Densities.txt', usecols=(1))

def getSource(energyRange):
    for i, source in enumerate(sourceList):
        if energyRange[0] >= sourceList[source][0] and energyRange[1] <= sourceList[source][1]:
            return source
    assert False, "Energy range too large for available datasets."

def getCRLSource(energyRange):
    for i, source in enumerate(sourceCRLList):
        if energyRange[0] >= sourceCRLList[source][0] and energyRange[1] <= sourceCRLList[source][1]:
            return source
    assert False, "Energy range too large for available datasets."

def initializeMaterial(mat, energyRange, returnSource=False, database='none'):
    assert database == 'none' or database in sourceList.keys(), "Database not found."
    if database == 'none':
        source = getSource(energyRange)
    else:
        assert sourceList[database][0] <= energyRange[0] and sourceList[database][1] >= energyRange[1], "Energy range out of bounds for given database."
        source = database
    data = getMat(mat, source)
    energy = data[:,0] # keV
    linearAttenuation = data[:,1] # cm-1
    # Different sources store data in different units
    if source == 'CXRO':
        energy = energy / 1000 # eV --> keV
    elif source == 'NIST':
        linearAttenuation = 1/linearAttenuation*(10**4) # cm-1 --> um
    else:
        assert False
    
    # Find slicing indexes to crop data down to only what you care about
    idxLower = (np.abs(energy - energyRange[0])).argmin() - 1
    idxUpper = (np.abs(energy - energyRange[1])).argmin() + 1
    # Prevent errors in slicing
    idxLower = 0 if idxLower < 0 else idxLower
    idxUpper = energy.shape[0] if idxUpper > energy.shape[0] else idxUpper
    
    energy = energy[idxLower:idxUpper]
    linearAttenuation = linearAttenuation[idxLower:idxUpper]
    # 'linear' instead of 'cubic' because there are discontinuities in the
    # x-ray cross-section of materials and the cubic functions struggle with these
    interpolation = interp1d(energy, linearAttenuation, kind='linear')
    if returnSource:
        return lambda x: interpolation(x), source
    else:
        return lambda e, d: np.exp(-d / interpolation(e))

def initializeLensMaterial(mat, energyRange, returnSource=False, database='none'):
    assert database == 'none' or database in sourceCRLList.keys(), "Database not found."
    if database == 'none':
        source = getCRLSource(energyRange)
    else:
        assert sourceCRLList[database][0] <= energyRange[0] and sourceCRLList[database][1] >= energyRange[1], "Energy range out of bounds for given database."
        source = database
    data = getCRLMat(mat, source)
    energy = data[:,0] # keV
    decriment = data[:,1] # unitless
    # Different sources store data in different units
    if source == 'Composite':
        energy = energy / 1000 # eV --> keV
    #elif source == 'NIST':
    #    linearAttenuation = 1/linearAttenuation*(10**4) # cm-1 --> um
    else:
        assert False
    
    # Find slicing indexes to crop data down to only what you care about
    idxLower = (np.abs(energy - energyRange[0])).argmin() - 1
    idxUpper = (np.abs(energy - energyRange[1])).argmin() + 1
    # Prevent errors in slicing
    idxLower = 0 if idxLower < 0 else idxLower
    idxUpper = energy.shape[0] if idxUpper > energy.shape[0] else idxUpper
    
    energy = energy[idxLower:idxUpper]
    decriment = decriment[idxLower:idxUpper]
    interpolation = interp1d(energy, decriment, kind='linear')
    if returnSource:
        return lambda x: interpolation(x), source
    else:
        return lambda e: interpolation(e)

def returnValidMaterials():
    return validMaterials

def returnValidSources():
    return sourceList.keys()

def updateDatabase():
    # read files from database and ensure they go on valid materials.
    pass

def convert_weightFraction_to_linearFraction(compList, weightFraction):
    all_densities = getDensities()
    densList = np.zeros(len(compList))
    for i in range(len(compList)):
        densList[i] = all_densities[validMaterials.index(compList[i])]
    linearFraction = np.array(weightFraction) / densList
    linearFraction /= np.sum(linearFraction)
    return linearFraction.tolist()