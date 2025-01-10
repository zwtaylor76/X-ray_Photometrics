# Import needed modules here, probably scipy and numpy
import numpy as np
import Photometrics.FetchMaterials as FetchMaterials
from typing import Dict, Tuple, List, Union, Callable
import functools
from operator import mul
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, root_scalar
import warnings
from collections.abc import Iterable

# It may be more useful to define some of these as lambda functions instead
# of classes. I will need to think about this.

# Note to self:
    # scipy.optimize.minimize(fun, x0, args=(), method=None, bounds=None, constraints=())
    # where fun(x0, *args) with args from optimizer
    # args is all fixed values
    # but you can also just pass everything in x0 and say make bounds a constant
    # bounds = [(min, max), (a,a), ...] where (a,a) means the value is fixed
    #
    # Can also use constraints if we need a more complex function
    # Ex: NonlinearConstraint(con, -np.inf, 1.9) with con = lambda x: x[0] - np.sin(x[1])

class Sample():
    # This class allows you to calculate the transmission as a function of
    # energy and thicknesses parametrizing a multi-material structure.
    #
    # Example Use Case:
        # sample_structure = ['Al', 'Ti', 'Ta']
        # energy_range = (8, 80) # keV
        # sample = Sample(sample_structure, energy_range)
        #
        # # Transmission at 66 keV through 80um Al + 10um Ti + 5um Ta
        # sample.transmission(66, [80, 10, 5])
        #
        # # Plot transmission vs energy for this set of thicknesses
        # e_range = np.arange(energy_range[0], energy_range[1], 0.1)
        # plt.plot(e_range, sample.transmission(e_range, [80, 10, 5]))
        #
        # # Plot transmission vs thickness of Ta at 50 keV, holding the other materials constant
        # thickness_range = np.arange(1, 10, 1)
        # transmissions = list(map(lambda t: sample.transmission(50, [80, 10, t]), thickness_range))
        # plt.plot(thickness_range, transmissions)
    #
    # Note, there is a soft failure if thickness list of incorrect size is passed
    # into the sample.transmission(e, [thickness])
    #
    # Further, one nuance is that this transmission is based on the linear
    # attenuation coefficient. So if you are approximating alloys from the pure
    # materials, this model would take in the volumetric (linear) fraction
    # rather than the mass or mole fraction.
    #
    # Interfaces with the FetchMaterials.py file to collect data files pulled
    # from CXRO and NIST (stored locally). This file returns transmission
    # functions vs energy and scalar thickness.
    #
    valid_materials = FetchMaterials.validMaterials

    def __init__(self, materialList: List[str], 
                 energyRange: Tuple[float, float], 
                 densityScaling : List[float] = None,
                 forceDataset: str = 'none'):
        # Take a list of materials: note thicknesses will be evaluated in this order
        # Take a tuple of photon energy range (min energy, max energy)
        if densityScaling is None:
            densityScaling = np.ones(len(materialList))
        else:
            assert len(densityScaling) == len(materialList), "densityScaling and materialList should be the same length."
            
        # FetchMaterials will throw the error if the material is not in the database
        # Get a list of Transmission Functions vs Energy & Thickness
        transmissivity = list(map(lambda kv: FetchMaterials.initializeMaterial(kv, energyRange, database=forceDataset), materialList))
        
        # A function of e, [Thickness List] returning the product of the transmissions of each material with that thickness
        self.transmission = lambda e, thickness: functools.reduce(mul, [transmissivity[i](e, t*densityScaling[i]) for i, t in enumerate(thickness)], 1)
        self.energy_range = energyRange
        self.materialList = materialList
        self.densityScaling = densityScaling
    
    # Function that allows you to define an interpolation of any material property
    # as needed, though intended to be of the form property vs energy where
    # materialProperty[:,0] is energies and materialProperty[:,1] is the property.
    def make_material_property(materialProperty : np.ndarray):
        assert len(materialProperty.shape) == 2, "Incorrect dimensionality of material property, needs to be 2."
        assert materialProperty.shape[1] == 2, "Needs to be of shape, (n, 2)."
        interpolation = interp1d(materialProperty[:, 0], materialProperty[:, 1], kind='linear')
        return lambda e : interpolation(e)

    def convert_weightFraction_to_linearFraction(self, weightFraction : List[float]):
        return FetchMaterials.convert_weightFraction_to_linearFraction(self.materialList, weightFraction)

class Scintillator(Sample):
    # A scintillator is defined by its transmission/attenuation, as with a normal sample,
    # but it also has a conversion efficiency
    #
    # Material = str, 'material'
    # Thickness in [um]
    # Conversion in visible photons per keV of input xray
    # Energy Range in keV
    def __init__(self, material: str, 
                 thickness: float, 
                 conversion: Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]], 
                 energyRange: Tuple[float, float]):
        Sample.__init__(self, [material], energyRange)
        self.absorption = lambda e: 1.0 - self.transmission(e, [thickness])
        
        if type(conversion) is float or type(conversion) is int:
            self.conversion = lambda e : conversion
        elif type(conversion) is np.ndarray:
            assert len(conversion.shape) == 2, "Incorrect dimensionality of supplied conversion list"
            self.conversion = Sample.make_material_property(conversion)
        else:
            self.conversion = conversion
        
        self.emission = lambda e: self.absorption(e) * self.conversion(e)
    
    ##########################################################################
    # Maybe define depth of focus, circle of confusion, and other stuff?
    ##########################################################################
    

class LensMaterial(Sample):
    def __init__(self, material: str,
                 thickness: float,
                 energyRange: Tuple[float, float],
                 forceDataset: str = 'none'):
        Sample.__init__(self, [material], energyRange)
        self.absorption = lambda e: 1.0 - self.transmission(e, [thickness])
        self.decriment = FetchMaterials.initializeLensMaterial(material, energyRange, database=forceDataset)

# Functions for geometric clipping calculations which are needed by multiple classes.
def circle_overlap_onto_rectangle(db : float, hc : float, wc : float):
    if db <= np.min([hc, wc]):
        beam_captured_by_camera = 1.0
        camera_filled_by_beam = (np.pi*(db/2)**2) / (hc*wc + 1e-25)
    elif db >= np.sqrt(hc**2 + wc**2):
        camera_filled_by_beam = 1.0
        beam_captured_by_camera = (hc*wc) / (np.pi*(db/2)**2 + 1e-25)
    elif db >= np.min([hc, wc]) and db <= np.max([hc, wc]):
        h = (db - np.min(hc, wc)) / 2
        r = db/2
        diff = r**2 * np.arccos(1 - h/r) - (r-h)*np.sqrt(r**2 - (r-h)**2)
        diff *= 2
        area = np.pi * r**2
        beam_captured_by_camera = (area - diff) / (area + 1e-25)
        camera_filled_by_beam = (area - diff) / (hc*wc + 1e-25)
    else:
        h1 = (db - hc) / 2
        h2 = (db - wc) / 2
        r = db/2
        diff1 = r**2 * np.arccos(1 - h1/r) - (r-h1)*np.sqrt(r**2 - (r-h1)**2)
        diff1 *= 2
        diff2 = r**2 * np.arccos(1 - h2/r) - (r-h2)*np.sqrt(r**2 - (r-h2)**2)
        diff2 *= 2
        area = np.pi * r**2
        beam_captured_by_camera = (area - diff1 - diff2) / (area + 1e-25)
        camera_filled_by_beam = (area - diff1 - diff2) / (hc*wc + 1e-25)
    
    return beam_captured_by_camera, camera_filled_by_beam

def rectangle_overlap_onto_rectangle(hb : float, wb : float, hc : float, wc : float):
    beam_captured_by_camera = np.min([hb, hc]) * np.min([wb, wc]) / (hb*wb + 1e-25)
    camera_filled_by_beam = np.min([hb, hc]) * np.min([wb, wc]) / (hc*wc + 1e-25)
    return beam_captured_by_camera, camera_filled_by_beam

def circle_overlap_onto_circle(db : float, dc : float):
    ratio = ((db + 1e-25) / (dc + 1e-25))**2
    beam_captured_by_camera = np.minimum(1, 1/ratio)
    camera_filled_by_beam = np.minimum(1, ratio)
    return beam_captured_by_camera, camera_filled_by_beam

def rectangle_overlap_onto_circle(hb : float, wb : float, dc : float):
    # Should be the same as the symmetric case above, but reversing the inputs and the outputs.
    camera_filled_by_beam, beam_captured_by_camera = circle_overlap_onto_rectangle(dc, hb, wb)
    return beam_captured_by_camera, camera_filled_by_beam

# Back to class definitions.
class Detector():
    # Direct:
        # Pixel size
        # Pixel count
        # Efficiency as a function of energy
        # Frame rate (for radiography experiments)
    # Indirect:
        # Everything in Direct, plus
        # Scintillator size
        # Scintillator material
        # Scintillator attentuation
        # Scintillator conversion efficiency
        # Imaging lens
        # Scintillator-lens-camera positions
            # Optimize if not provided
            # Report if image cannot be formed, or if provided but wrong
    def __init__(self, direct : bool, 
                 pixelSize : float, 
                 pixelCount : Tuple[float, float],
                 pixelEfficiency : Union[float, np.ndarray, Tuple[Sample, List[float]], Callable[[np.ndarray], np.ndarray]], 
                 exposureTime : float,
                 scintillator : Scintillator = None, 
                 scintillatorSize : float = None, 
                 opticalMagnification : float = None,
                 lensFocalLength : float = None,
                 lensNumericalAperture : float = None,
                 scintillator_lens_camera_position : Tuple[float, float] = None):
        if direct:
            assert scintillator is None, "Scintillator should not be present in direct detection mode."
            assert scintillatorSize is None, "Scintillator should not be present in direct detection mode."
            assert lensFocalLength is None, "No optical lens is present in direct detection mode."
            assert opticalMagnification is None, "No optical magnification is present in direct detection mode."
            assert lensNumericalAperture is None, "No optical lens is present in direct detection mode."
            assert scintillator_lens_camera_position is None, "No optical camera is present in direct detection mode."
            self.energy_range = None
        else:
            # Scintillator material
            self.scintillator = scintillator
            # Diameter [um], assuming circle
            self.scintillatorSize = scintillatorSize
            self.energy_range = scintillator.energy_range
            
            if opticalMagnification is None:
                assert lensFocalLength is not None, "Must provide lens focal length for indirect detection mode, or an opticalMagnification."
                assert scintillator_lens_camera_position is not None, "Must provide lens positions for indirect detection mode, or an opticalMagnification."
                assert lensNumericalAperture is not None, "Must provide lens numerical aperture for indirect detection mode."
                self.calculateMagnification = True
                # Focal length in [m]
                self.lensFocalLength = lensFocalLength
                # NA Numerical Aperture
                self.lensNumericalAperture = lensNumericalAperture
                # (Scintillator to lens length, lens to camera detector length) in [m]
                self.scintillator_lens_camera_position = scintillator_lens_camera_position
            else:
                assert lensFocalLength is None, "General magnification overides lens focal length."
                assert scintillator_lens_camera_position is None, "General magnification overides optical parameters."
                assert lensNumericalAperture is not None, "Numerical aperture of lens (or equivalent calculated from aperture size) must be provided in indirect detection mode."
                # Even if just optical magnification is provided, an aperture is needed
                self.calculateMagnification = False
                self.opticalMagnification = opticalMagnification
                # NA (Numerical Aperture)
                self.lensNumericalAperture = lensNumericalAperture
        
        # Direct detection vs indirect detection
        self.direct = direct
        # Pixel size in [um]
        self.pixelSize = pixelSize
        # Number of pixels on detector (h,w)
        self.pixelCount = pixelCount
        # Exposure time [s], often 1/frame rate
        self.exposureTime = exposureTime
        
        # Efficiency of detection of a pixel given as:
        # Float: one efficiency for all wavelengths
        if type(pixelEfficiency) is float:
            self.pixelEfficiency = lambda e : pixelEfficiency
        # np.ndarray: numpy array of efficiencies
        elif type(pixelEfficiency) is np.ndarray:
            assert len(pixelEfficiency.shape) == 2, "Incorrect dimensionality of supplied efficiency list"
            self.pixelEfficiency = Sample.make_material_property(pixelEfficiency)
            if self.energy_range is None:
                self.energy_range = (np.min(pixelEfficiency[0]), np.max(pixelEfficiency[0]))
        # Sample: returns transmissions as a function of energy and thickness
        elif type(pixelEfficiency) is tuple and type(pixelEfficiency[0]) is Sample and type(pixelEfficiency[1]) is list:
            # Effiency can be approximated as 1 - transmission = absorption
            self.pixelEfficiency = lambda e : 1 - pixelEfficiency[0](e, pixelEfficiency[1])
            if self.energy_range is None:
                self.energy_range = pixelEfficiency[0].energy_range
        # Callable: returns efficiency as a function of energy
        else:
            self.pixelEfficiency = pixelEfficiency
        #else:
        #    print(type(pixelEfficiency))
        #    raise(Exception("Incorrect type given for pixelEfficiency"))
    
    def calculateOpticalMagnification(self):
        if self.calculateMagnification:
            # Verify that image can be formed
            necessaryFocalLength = 1 / (1/self.scintillator_lens_camera_position[0] + 1/self.scintillator_lens_camera_position[1])
            if necessaryFocalLength == self.lensFocalLength:
                return self.scintillator_lens_camera_position[1] / self.scintillator_lens_camera_position[0]
            else:
                assert False, f"Image Cannot be Formed: try focal-length = {necessaryFocalLength} or vary scintillator-lens-camera positions."
        else:
            return self.opticalMagnification
    
    def calculateGuage(self, mag : float):
        if type(self.pixelSize) is tuple:
            return (self.pixelSize[0]/mag, self.pixelSize[1]/mag)
        else:
            return self.pixelSize/mag
    
    def calculateIndirectAngularEfficiency(self):
        if self.direct:
            assert False, "A direct detector has no indirect detection efficiency."
        else:
            if self.calculateMagnification:
                aperture = self.calculate_aperture_from_NA()
                return (aperture / self.scintillator_lens_camera_position[0])**2 / 16
            else:
                print('Warning: underdefined acceptance angle from scintillator into camera lens.')
                theta = np.arcsin(self.lensNumericalAperture)
                return (1 - np.cos(theta))/2
    
    def calculate_aperture_from_NA(self):
        return 2 * self.lensFocalLength * self.lensNumericalAperture
    
    def fluxDensity(self, density_at_xray_image_plane : Callable[[float], float]):
        if self.direct:
            return density_at_xray_image_plane
        else:
            return lambda e : density_at_xray_image_plane(e) / self.opticalMagnification**2

class Source():
    # Divergence
        # As angle from collimated. Include parameter for 'collimated'==0.0
    # Spot size
    # Flux density
        # Requires a distribution of flux vs energy within the energy bounds of material
        # function for returning Dirac delta at a specific energy
        # If scalar, uniform distribution over all energies
        # THe ability to autoconvert between energies and photon counts
    # Source-to-sample distance (or optimized if not found)
        # function for flux density and spot size as a function of distance from source
    # Spatial distribution of beam? Only useful in conjunction with Laura's stuff.
    # Maybe the ability to include upstream optics?
    # In short, everything upstream of the sample.
    
    collimated = 0.0
    
    def __init__(self, divergence : float, 
                 spotSize : Union[float, Tuple[float, float]], 
                 spectrum : Union[Tuple[np.ndarray, Tuple[str, None]], Tuple[Callable[[float], float], Tuple[str, None]]], #Callable[[float, float], float], ? # np.ndarray, Callable[[float], float], 
                 spectrum_pulsed : bool, source_to_sample : float = None, 
                 spatialDistribution : np.ndarray = None):
        # Angle of the beam [rad]:
            # Negative is converging
            # Zero is collimated
            # Positive is diverging
        self.divergence = divergence
        
        # Spectrum assumed constant accross all pixels/spot size
        if type(spectrum[0]) is np.ndarray:
            assert len(spectrum[0].shape) == 2, "Incorrect dimensionality of supplied spectrum list"
            self.spectrum = Sample.make_material_property(spectrum[0])
            self.energy_range = (np.min(spectrum[0][:, 0]), np.max(spectrum[0][:, 0]))
            self.spectrum_type = spectrum[1]
            #print(self.spectrum_type)
        else:
            self.spectrum = spectrum
            self.energy_range = None
            self.spectrum_type = spectrum[1]
            #print(self.spectrum_type)
        # If True, spectrum in ph/pulse vs keV
        # If False, spectrum in ph/s vs keV
        self.pulsed = spectrum_pulsed
        
        if type(spotSize) == float and spatialDistribution is None:
            # Circle of diameter 'spotSize' [um]
            self.spotShape = 'circle'
            self.spotSize = spotSize
        elif spatialDistribution is None:
            # Rectangle of (h,w) = spotSize [um]
            self.spotShape = 'rectangle'
            self.spotSize = spotSize
        else:
            # A 2D ndarray with (h,w) dimensions = spotSize [um]
            # Each value in ndarray denotes the relative brightness of that spot as a fraction of the whole
            self.spotShape = 'custom'
            assert type(spotSize) is float, "For a spatial distribution, spotSize must be a float defining 'pixel' size."
            self.spotSize = spotSize
            # Ensures proper normalization & unless everything is zero (epsilon for numerical stability)
            self.shape = spatialDistribution / (np.sum(spatialDistribution) + 1e-25)
        
        #########################################################################
        # Change later so that source_to_sample can be solved for if not provided
        #########################################################################
        assert source_to_sample is not None
        # Source to sample distance in [m]
        self.source_to_sample = source_to_sample
    
    def fetchSourceSpectrum(filepath : str, scaling : int = 1.0, pad : bool = False, ev : bool = False, threshold : float = 0.0, clip : Tuple[float, float] = None):
        spec = np.loadtxt(filepath)
        spec[:, 1] *= scaling
        if ev:
            spec[:, 0] /= 1000
        #if pad:
        if clip is None and pad:
            zerosBefore = np.array([[0, 0], [spec[0, 0]-1e-5, 0]])
            zerosAfter = np.array([[spec[-1, 0]+1e-5, 0], [400., 0]])
            spec = np.concatenate((zerosBefore, spec, zerosAfter))
        else:
            lower_bound = np.argmin(np.abs(spec[:,0] - clip[0]))
            if lower_bound < 0:
                spec = np.concatenate((np.array([[clip[0], 0], [spec[0, 0]-1e-5, 0]]), spec))
            else:
                spec = spec[lower_bound:]
                spec[0,1] = (spec[0,1] - spec[1,1])/(spec[0,0] - spec[1,0])*(spec[0,0] - clip[0])
                spec[0,0] = clip[0]
            
            upper_bound = np.argmin(np.abs(spec[:,0] - clip[1])) + 2
            #upper_bound = upper_bound if spec[upper_bound,0] > clip[1] else upper_bound + 1
            if upper_bound > spec.shape[0]:
                spec = np.concatenate((spec, np.array([[spec[-1, 0]+1e-5, 0], [clip[1], 0]])))
            else:
                spec = spec[:upper_bound]
                spec[-1,1] = (spec[-2,1] - spec[-1,1])/(spec[-2,0] - spec[-1,0])*(spec[-2,0] - clip[1])
                spec[-1,0] = clip[1]
                    
        counts = spec[:, 1]
        counts[counts < threshold] = 0.0
        spec[:, 1] = counts
        
        return spec, ("Import", filepath)
    
    ##########################################################################
    # Need to somehow make energy range implicit from sample?
    # Don't, let Experiment object verify that they are compatable.
    ##########################################################################
    def uniformSpectrum(flux : float, a : float, b : float, energy_range : Tuple[float, float] = None):
        assert b > a, "Lower bound on energy greater than upper bound."
        if energy_range is not None:
            assert a >= energy_range[0], "Spectrum energy range is not the same as the material energy range."
            assert b <= energy_range[1], "Spectrum energy range is not the same as the material energy range."
        return (lambda x: flux/(b-a) if (x <= b and x >= a) else exec('raise(Exception("Energy outside the range of the spectrum."))'), ("Const", (a, b)))
    
    def monochromaticSpectrum(flux : float, e : float, energy_range : Tuple[float, float] = None):
        if energy_range is not None:
            assert e >= energy_range[0], "Spectrum energy is not within the material energy range."
            assert e <= energy_range[1], "Spectrum energy is not within the material energy range."
        #return lambda x: flux if x==e else 0.0
        return lambda x : flux*(x==e), ("Mono", e)
        ######################################################################
        # Figure out how to regist monochromatic energy so that during the
        # photon count numerical integration in the last step, the monochromatic
        # peak isn't missed?
        ######################################################################
    
    ##########################################################################
    # State functions for returning illumination size at sample
    # plus whatever is needed by the other functions to calculate the end properties
    # plus whatever is needed for optimization
    ##########################################################################
    def dilation(self, length : float):
        return length * np.tan(self.divergence)
    
    def fluxDensity(self, length : float):
        # return photons / um^2 / s (pulse) as a function of distance from source definition and energy
        o = self.dilation(length)
        if self.spotShape == 'circle':
            area = np.pi * (self.spotSize/2 + o)**2
            if type(self.spectrum) is tuple:
                return lambda e : self.spectrum[0](e) / area
            else:
                return lambda e : self.spectrum(e) / area
        elif self.spotShape == 'rectangle':
            area = np.abs((self.spotSize[0] + 2*o)*(self.spotSize[1] + 2*o))
            if type(self.spectrum) is tuple:
                return lambda e : self.spectrum[0](e) / area
            else:
                return lambda e : self.spectrum(e) / area
        elif self.spotShape == 'custom':
            area = np.abs((self.spotSize + 2*o)*(self.spotSize + 2*o))
            if type(self.spectrum) is tuple:
                return lambda e : self.spectrum[0](e) / area * self.shape
            else:
                return lambda e : self.spectrum(e) / area * self.shape
        else:
            assert False, "Incorrect beam profile."
       
class Microscope():
    # Radiography vs TXM
    # All geometric constraints:
        # L
        # d1, T, d2
        # CRL vs Bragg
        # Materials (attenuation, magnification, etc.)
        # + functions from Mathematica implementing Simons paper
    # Sample-to-detector distance (or optimized if not found)
    # In short, everything downstream of the sample.
    validParameters = {'radiograph' : ['sample_to_detector'],
                       'CRL' : ['material', 'N', 'radius', 'aperture', 'lens_thickness', 'sample_to_detector', 'sample_to_lens', 'lens_to_detector'],
                       'bragg' : ['material', "cubic_lattice_constant_a", "angular_acceptance", "N-bounce"]}
    
    def __init__(self, microscopeType : str, 
                 definingParameters : Dict[str, Union[int, float, LensMaterial]]):
        assert microscopeType in Microscope.validParameters.keys(), "Invalid microscope type."
        assert set(list(definingParameters)) == set(Microscope.validParameters[microscopeType]), "Improper keys for given microscope type."
        
        self.microscopeType = microscopeType
        # Radiography
        if microscopeType == 'radiograph':
            # [m]
            self.sample_to_detector = definingParameters['sample_to_detector']
        # TXM with CRL
        elif microscopeType == 'CRL':
            # CRL-based calculations pulled from Simons 2017, "Simulating and Optimizing CRL-based X-ray Microscopes"
            self.material = definingParameters['material']
            self.N = definingParameters['N']
            # [m]
            self.radius = definingParameters['radius']
            # [m]
            self.aperture = definingParameters['aperture']
            # [m]
            self.sample_to_detector = definingParameters['sample_to_detector']
            # [m]
            self.sample_to_lens = definingParameters['sample_to_lens']
            # [m]
            self.lens_thickness = definingParameters['lens_thickness']
            # [m]
            self.lens_to_detector = definingParameters['lens_to_detector']
            
            assert type(self.material) is LensMaterial, "material must be a 'LensMaterial'."
            assert type(self.N) is int, "N must be an int."
            assert type(self.radius) is float, "radius must be a float."
            assert type(self.sample_to_detector) is float, "sample_to_detector must be a float."
            assert self.sample_to_lens is None or type(self.sample_to_lens) is float, "sample_to_lens must be a float."
            assert type(self.lens_thickness) is float, "lens_thickness must be a float."
            assert self.lens_to_detector is None or type(self.lens_to_detector) is float, "lens_to_detector must be a float."
            
            calculateD1 = False
            calculateD2 = False
            if self.sample_to_lens is None:
                calculateD1 = True
            if self.lens_to_detector is None:
                calculateD2 = True
            
            if not calculateD2:
                if not calculateD1:
                    assert self.sample_to_detector == self.N*self.lens_thickness + self.sample_to_lens + self.lens_to_detector, "Sample geometry does not align."
                    self.magnification = self.magnification_CRL_D2()
                else:
                    self.magnification = self.magnification_CRL_D2()
                    self.sample_to_lens = self.calculate_D1()
            else: #calculate D1
                self.magnification = self.magnification_CRL_L()
                self.sample_to_lens, self.lens_to_detector = self.calculate_D1D2()
            
            self.f = lambda e : self.radius / (2 * self.material.decriment(e))
            self.phi = lambda e : np.sqrt(self.lens_thickness / self.f(e))
            self.fN = lambda e : self.f(e) * self.phi(e) / np.tan(self.N * self.phi(e))
        # Holography with Bragg Magnifier
        elif microscopeType == 'bragg':
            # Bragg magnifier calculations pulled from 
                # Vagovic 2011, "In-line Bragg magnifier based on V-shaped germanium crystals"
                # Spal 2001, "Submicron resolution hard x-ray holography with asymmetric bragg diffraction microscope"
                # plus basic trig identities
            assert type(definingParameters['material']) is LensMaterial, "Magnifier crystal must be a LensMaterial"
            assert type(definingParameters['cubic_lattice_constant_a']) is float, "A cubic crystal is assumed with lattice constant a, float in angstroms"
            assert callable(definingParameters['angular_acceptance']), "A function defining angular acceptance of a Bragg diffraction, rad"
            assert type(definingParameters['N-bounce']) is int, "N-bounce must be int"
            assert definingParameters['N-bounce'] in [1,2,4], "1 = 1-bounce 1D mag, 2 = 2-bounce 1D mag, 4 = 4-bounce 2D mag"
            
            self.material = definingParameters['material']
            self.a = definingParameters['cubic_lattice_constant_a']
            self.acceptance = definingParameters['angular_acceptance']
            self.N_bounce = definingParameters['N-bounce']
            
            self.wavelength = lambda e : 12.398/e
            # Wavelength in A of an x-ray with energy keV
            self.dSpacing = lambda hkl : self.a / np.sqrt(hkl[0]**2 + hkl[1]**2 + hkl[2]**2)
            # Interplanar spacing of a cubic crystal
            
            self.gamma = lambda e, hkl : self.wavelength(e) / (2*self.dSpacing(hkl))
            # sin(theta) in Bragg's Law
            self.theta = lambda e, hkl : np.arcsin(self.gamma(e, hkl))
            # Bragg's law, solving for theta, the Bragg angle of diffraction
            
            self.magnification = lambda e, hkl, miscut : 1 + (self.gamma(e, hkl)*np.sqrt(1-(self.gamma(e, hkl))**2)*np.sin(2*miscut))/((self.gamma(e, hkl))**2-np.sin(miscut)**2)
            # Bragg magnification (ignoring refractive correction)
            
            self.theta_c_TER = lambda e : np.sqrt(2*self.material.decriment(e))
            # Critical angle for total external reflection (TER) in vacuum
            # If (theta - alpha) < TER, then the bragg magnifier will not work because it reflects instead of diffracts
            
            self.magnification_maximum = lambda e, hkl, scale : np.array(list(map(lambda ei : self.magnification(ei, hkl, self.theta(ei, hkl - scale*self.theta_c_TER(ei))), e)))
            # According to Spal 2001, scale = sqrt(2)
            # This is the measure of how close you can get to total external reflection without causing problems
            
            # Calculate needed magnifications, flux decays for different bounce cases
            # Note, this assumes a collimated beam in
            if self.N_bounce == 1:
                self.totalXMag = self.magnification
                self.totalXMag_maximum = self.magnification_maximum
                self.flux_correction = lambda e, hkl, miscut : 1.0 / self.totalXMag(e, hkl, miscut)
            
            elif self.N_bounce == 2:
                self.totalXMag = lambda e, hkl, miscuts : self.magnification(e, hkl, miscuts[0]) * self.magnification(e, hkl, miscuts[1])
                self.totalXMag_maximum = lambda e, hkl, scale : self.magnification_maximum(e, hkl, scale)**2
                self.flux_correction = lambda e, hkl, miscuts : 1.0 / self.totalXMag(e, hkl, miscuts)
            
            elif self.N_bounce == 4:
                self.totalXMag_1D = lambda e, hkl, miscuts : self.magnification(e, hkl, miscuts[0]) * self.magnification(e, hkl, miscuts[1])
                self.totalXMag_maximum_1D = lambda e, hkl, scale : self.magnification_maximum(e, hkl, scale)**2
                self.flux_correction = lambda e, hkls, miscuts : 1.0 / self.totalXMag_maximum_1D(e, hkls[0], (miscuts[0], miscuts[1])) / self.totalXMag_maximum_1D(e, hkls[1], (miscuts[2], miscuts[3]))
                
            self.bandwidth = lambda e, hkl : (1.0 / self.acceptance(hkl)) * e / np.sqrt(self.gamma(e, hkl)**-2 - 1.0)
            self.dirac = lambda e, e_ : (e == e_)*1.0
            
            ##################################################################
            # In the future: 
                # Add refractive correction
                # Calculate diffraction limit of resolution from angular acceptance (for Bragg as well as TXM)
                # Calculate efficiency of diffraction
                # Calculate angular acceptance within code
            ##################################################################

        else:
            assert False, "Incorrect microscope type."
        
        ######################################################################
        # Write code so some parameters do not need to be provided if
        # we expect to be optimizing.
        ######################################################################
    
    def magnification_CRL_D2(self):
        d2 = self.lens_to_detector
        magnifications = lambda e : (d2 / (self.f(e) * self.phi(e))) * np.sin(self.N * self.phi(e)) - np.cos(self.N * self.phi(e))
        return magnifications
    
    def calculate_D1(self):
        d1 = lambda e : self.fN(e) * (1 + 1 / (self.magnification(e) * np.cos(self.N * self.phi(e))))
        self.sample_to_lens = d1
        # Flag so that if you ever need to use d1, you will know it's been calculated as a function.
        self.d1_func = True
        return d1
    
    def printNote(value : float, threshold : float, warn : bool = True):
        if value < threshold and warn:
            print('Note: At this photon energy and microscope geometry, an image cannot be formed.')
        return value
    
    def crl_func(self, M, e):
        # Objective function for minimization, from Simons paper
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            lnt = self.sample_to_detector - self.N*self.lens_thickness
            val = (1 / self.phi(e)) * (np.arcsin((M + 1/M) / np.sqrt(4 + ((lnt)/(self.f(e)*self.phi(e)))**2)) + np.arctan((2*self.f(e)*self.phi(e)) / (lnt)))
        
        return val
    
    def magnification_CRL_L(self):
        # with warnings.catch_warnings():
            # Note: there will occasionally be a RuntimeWarning raised when the
            # arcsin is evaluated, since the guessed magnification can be too 
            # large for the domain of the arcsin function. This problem could
            # later be solved by replacing the 100 in the bracket of the minimize_scalar
            # with the maximum value of M that satisfies the argument of arcsin
            # being less than 1. But this is more trouble than it's worth right now.
            # func = lambda M, e : (1 / self.phi(e)) * (np.arcsin((M + 1/M) / np.sqrt(4 + ((lnt)/(self.f(e)*self.phi(e)))**2)) + np.arctan((2*self.f(e)*self.phi(e)) / (lnt)))
                
                # For purposes warning suppression, "func" had to be moved to its own
                # function instead of being used as a lambda function
            
            # Golden optimization method equivalent to bisection for root-finding
            # using (1, 100) bracked should ensure calculation of magnification solution
            # there also exists a demagnification solution, but this is not what is generally desired
            result = lambda e : minimize_scalar(lambda M : np.nan_to_num((self.N - self.crl_func(M, e))**2, nan=np.inf), bracket=(1, 100), method='golden').x
            #result = lambda e : minimize_scalar(lambda M : (self.N - self.crl_func(M, e))**2, bounds=(1, 100), method='bounded').x
                # Performs well
            
            return lambda e : Microscope.unmakeIterable(np.array(list(map(lambda ei : result(ei), Microscope.makeIterable(e)))))
            # This catches the case in which single numbers are passed in.
            # Normally, this case is handled all the way down at the interpolation,
            # but for some reason, lists wasn't working here in the interpolation, 
            # so I had to do this list mapping to get both lists and numbers to
            # work for plotting purposes.
    
    def makeIterable(value):
        return value if isinstance(value, Iterable) else [value]
    
    def unmakeIterable(value):
        return value if value.shape != (1, ) else value[0]
    
    def calculate_D1D2(self):
        d1 = self.calculate_D1()
        d2 = lambda e : self.fN(e) * (1 + self.magnification(e) / (np.cos(self.N * self.phi(e))))
        return d1, d2
    
    def fluxDensity_crl(self, at_sample : bool, source : Source):
        flux_atSample = source.fluxDensity(source.source_to_sample)
        flux_atCRL = lambda e : source.fluxDensity(source.source_to_sample + self.sample_to_lens(e))(e)
        if at_sample:
            return flux_atSample
        else:
            # Calculates flux at the x-ray image plane
            m = lambda e : self.magnification(e)
            
            ### Flux Density does not care about clipping
            ### Defined by photons / um^2 / pulse
            #
            ## "Radiographic" flux density at the entrance to the CRL
            #o = source.dilation(source.source_to_sample + self.sample_to_lens)
            #
            #if source.spotShape == 'circle':
            #    # Aperture clipping at the aperature
            #    _, clipping_CRL_Aperture = circle_overlap_onto_circle(self.aperture, source.spotSize/2 + o)
            #    #clipping_CRL_Aperture = ((self.aperture)/(source.spotSize/2+o))**2
            #    #clipping_CRL_Aperture = np.min(1, clipping_CRL_Aperture)
            #elif source.spotShape == 'rectangle' or source.spotShape == 'custom':
            #    if source.spotShape ==  'rectangle':
            #        hb, wb = source.spotSize
            #        dc = self.aperature
            #    else:
            #        hb, wb = source.shape.shape
            #        hb *= source.spotSize
            #        wb *= source.spotSize
            #        dc = self.aperture
            #    clipping_CRL_Aperture, _ = rectangle_overlap_onto_circle(hb + 2*o, wb + 2*o, dc)
            #else:
            #    assert False, "Incorrect beam profile."
            #
            #return lambda e : flux_atSample(e) * clipping_CRL_Aperture / (m(e)**2)
            
            return lambda e : flux_atSample(e) / (m(e)**2) * (1 - self.material.absorption(e))
            ##################################################################
            # Is this correct?
            # Should it be flux_atSample instead?
            # Or flux_atCRL
            ##################################################################

class Experiment():
    # Takes:
        # List of Samples
        # One Microscope
        # List of Detectors
        # One Source
    # Computes:
        # Attentuation
        # Photon flux (1/um or 1/px) at each detectors
        # Resolution and magnification for each detector
        # Fraction of detector illuminated 
        # Fraction of beam captured by detector
        # Allows for plotting of these values against any variable in any input
        # Allows for optimization of any of these values as a function of any subset of input variables
    #
    # May eventually want to interface this with Laura's simulation code
    # Dictionary so that you can name samples and detectors and plot them by their name
    def __init__(self, samples: Dict[str, Sample], 
                 microscope: Microscope, 
                 detectors: Dict[str, Detector], 
                 source: Source):
        self.samples = samples
        self.microscope = microscope
        self.detectors = detectors
        self.source = source
        
        # rename to 'self.precalculations' one day
        self.precalculations = {}
        
        # Verify that the energy ranges from the samples == those from detector scintillators
        # and are within the bounds of those from the source spectrum
        self.checkEnergyRanges()
        
        self.precalculate()
    
    def checkEnergyRanges(self):
        self.energyRanges = {}
        # needs to loop through samples and detectors
        for sampleName, sample in self.samples.items():
            for detectorName, detector in self.detectors.items():
                sampleEnergy = sample.energy_range
                sourceEnergy = self.source.energy_range
                detectorEnergy = detector.energy_range
                if sourceEnergy is not None:
                    assert sampleEnergy[0] <= sourceEnergy[0], "Sample energy lower bound above source energy lower bound."
                    assert sampleEnergy[1] >= sourceEnergy[1], "Sample energy upper bound below source energy upper bound."
                if detectorEnergy is not None:
                    assert sampleEnergy[0] <= detectorEnergy[0], "Sample energy lower bound above detector energy lower bound."
                    assert sampleEnergy[1] >= detectorEnergy[1], "Sample energy upper bound below detector energy upper bound."
                    if sourceEnergy is not None:
                        assert sourceEnergy[0] >= detectorEnergy[0], "Source energy lower bound below detector energy lower bound."
                        assert sourceEnergy[1] <= detectorEnergy[1], "Source energy upper bound above detector energy upper bound."
                
                if sourceEnergy is not None:
                    self.energyRanges[(sampleName, detectorName)] = sourceEnergy
                    self.precalculations[(sampleName, detectorName)] = sourceEnergy
                else:
                    if detectorEnergy is not None:
                        self.energyRanges[(sampleName, detectorName)] = detectorEnergy
                        self.precalculations[(sampleName, detectorName)] = detectorEnergy
                    else:
                        self.energyRanges[(sampleName, detectorName)] = sampleEnergy
                        self.precalculations[(sampleName, detectorName)] = sampleEnergy
    
    def precalculate(self):
        # needs to loop through samples and detectors
        for sampleName, sample in self.samples.items():
            for detectorName, detector in self.detectors.items():
                # Calculate flux densities at sample and image plane
                self.precalculations[(sampleName, detectorName)] = self.calculate_fluxDensities(sampleName)
                # Calculate flux densities at detector
                self.precalculations[(sampleName, detectorName)].update(self.calculate_photonCounts(sampleName, detectorName))
                # Calculate resolution and magnification at detector
                self.precalculations[(sampleName, detectorName)].update(self.calculate_resolutionMagnification(sampleName, detectorName))
                # Calculate beam and detector overlap
                self.precalculations[(sampleName, detectorName)].update(self.calculate_beamCameraOverlap(sampleName, detectorName))
               
    
    def calculate_fluxDensities(self, sampleName : str):
        if self.microscope.microscopeType == 'radiograph':
            fluxDensity_atSample = self.source.fluxDensity(self.source.source_to_sample)
            fluxDensity_atImage = self.source.fluxDensity(self.source.source_to_sample + self.microscope.sample_to_detector)
            fluxDensity_atImage_throughSample = lambda e, thickness : fluxDensity_atImage(e) * self.samples[sampleName].transmission(e, thickness)
            spec = {'fluxDensity_atSample' : fluxDensity_atSample,
                    'fluxDensity_atImage' : fluxDensity_atImage,
                    'fluxDensity_atImage_throughSample' : fluxDensity_atImage_throughSample,
                    'attenuation' : self.samples[sampleName].transmission}
            
            inc_sample = self.source.dilation(self.source.source_to_sample)
            inc_image = self.source.dilation(self.source.source_to_sample + self.microscope.sample_to_detector)
            if self.source.spotShape == 'circle':
                spec['spotSize_atSample'] = self.source.spotSize + 2*inc_sample
                spec['spotSize_atImage'] = self.source.spotSize + 2*inc_image
            elif self.source.spotShape == 'custom' or self.source.spotShape == 'rectangle':
                spec['spotSize_atSample'] = (self.source.spotSize[0] + 2*inc_sample, self.source.spotSize[1] + 2*inc_sample)
                spec['spotSize_atImage'] = (self.source.spotSize[0] + 2*inc_image, self.source.spotSize[1] + 2*inc_image)
            return spec
        
        elif self.microscope.microscopeType == 'CRL':
            fluxDensity_atSample = self.microscope.fluxDensity_crl(at_sample = True, source = self.source)
            fluxDensity_atImage = self.microscope.fluxDensity_crl(at_sample = False, source = self.source)
            fluxDensity_atImage_throughSample = lambda e, thickness : fluxDensity_atImage(e) * self.samples[sampleName].transmission(e, thickness)
            spec = {'fluxDensity_atSample' : fluxDensity_atSample,
                    'fluxDensity_atImage' : fluxDensity_atImage,
                    'fluxDensity_atImage_throughSample' : fluxDensity_atImage_throughSample,
                    'attenuation' : self.samples[sampleName].transmission}
            
            inc_sample = self.source.dilation(self.source.source_to_sample)
            #inc_image = self.source.dilation(self.source.source_to_sample + self.microscope.sample_to_detector)
            mag = self.microscope.magnification
            if self.source.spotShape == 'circle':
                spec['spotSize_atSample'] = self.source.spotSize + 2*inc_sample
                spec['spotSize_atImage'] = lambda e : spec['spotSize_atSample']*mag(e)
            elif self.source.spotShape == 'custom' or self.source.spotShape == 'rectangle':
                spec['spotSize_atSample'] = (self.source.spotSize[0] + 2*inc_sample, self.source.spotSize[1] + 2*inc_sample)
                spec['spotSize_atImage'] = (lambda e : spec['spotSize_atSample'][0]*mag(e), lambda e : spec['spotSize_atSample'][1]*mag(e))
            return spec
        
        elif self.microscope.microscopeType == 'bragg':
            fluxDensity_atSample = self.source.fluxDensity(self.source.source_to_sample)
            # Note that a Bragg magnifier is a monochromator
            # Here, e is the energy from spectrum, e_ is the energy we align the hkl for
            correction  = lambda e, e_, hkl : self.microscope.dirac(e, e_) * self.microscope.bandwidth(e, hkl)
            fluxDensity_atImage = lambda e, e_, hkl, miscut : fluxDensity_atSample(e) * self.microscope.flux_correction(e, hkl, miscut) * correction(e, e_, hkl)
            fluxDensity_atImage_throughSample = lambda e, e_, hkl, miscut, thickness : fluxDensity_atImage(e, e_, hkl, miscut) * self.samples[sampleName].transmission(e, thickness)
            spec = {'fluxDensity_atSample' : fluxDensity_atSample,
                    'fluxDensity_atImage' : fluxDensity_atImage,
                    'fluxDensity_atImage_throughSample' : fluxDensity_atImage_throughSample,
                    'attenuation' : self.samples[sampleName].transmission}
            
            # Spot size calculations adjusted for x vs y
            inc_sample = self.source.dilation(self.source.source_to_sample)
            if self.microscope.N_bounce == 1 or self.microscope.N_bounce == 2:
                # 1D magnification, which is a function of user inputs which could be different in x and y
                mag = self.microscope.totalXMag
                
                if self.source.spotShape == 'circle':
                    spec['spotSize_atSample'] = self.source.spotSize + 2*inc_sample
                    spec['spotSize_atImage'] = (spec['spotSize_atSample'], lambda e, hkl, miscut : spec['spotSize_atSample']*mag(e, hkl, miscut))
                elif self.source.spotShape == 'custom' or self.source.spotShape == 'rectangle':
                    spec['spotSize_atSample'] = (self.source.spotSize[0] + 2*inc_sample, self.source.spotSize[1] + 2*inc_sample)
                    spec['spotSize_atImage'] = (lambda e, hkl, miscut : spec['spotSize_atSample'][0]*mag(e, hkl, miscut), spec['spotSize_atSample'][1])
            elif self.microscope.N_bounce == 4:
                mag = self.microscope.totalXMag_1D
                
                if self.source.spotShape == 'circle':
                    spec['spotSize_atSample'] = self.source.spotSize + 2*inc_sample
                    spec['spotSize_atImage'] = lambda e, hkls, miscuts : spec['spotSize_atSample']*mag(e, hkls, miscuts)
                elif self.source.spotShape == 'custom' or self.source.spotShape == 'rectangle':
                    spec['spotSize_atSample'] = (self.source.spotSize[0] + 2*inc_sample, self.source.spotSize[1] + 2*inc_sample)
                    spec['spotSize_atImage'] = (lambda e, hkl, miscut : spec['spotSize_atSample'][0]*mag(e, hkl, miscut), lambda e, hkl, miscut : spec['spotSize_atSample'][1]*mag(e, hkl, miscut))
            else:
                assert False, "Only N-bounce = 1,2, or 4 is currently allowed"
                
            return spec
        else:
            assert False, "Microscope Type Not Programmed"
    
    def calculate_photonCounts(self, sampleName : str, detectorName : str):
        if self.microscope.microscopeType == 'radiograph':
            if self.source.spotShape == 'custom':
                ##################################################################
                # Need to figure out how custom beam profiles affect things
                # I think maybe just calculate photonCounts per pixel for each
                # index of the beam profile?
                # Either that or interpolate through space, but that seems unnecessary
                ##################################################################
                pass
            else:
                fluxDensity_atImage = self.precalculations[(sampleName, detectorName)]['fluxDensity_atImage']
                pixelArea = self.detectors[detectorName].pixelSize**2
                exposureTime = self.detectors[detectorName].exposureTime
                photons_per_pixel_perEnergy = lambda e : pixelArea * fluxDensity_atImage(e) * exposureTime
                
                if self.detectors[detectorName].direct:
                    photons_detected_per_pixel_perEnergy = lambda e : photons_per_pixel_perEnergy(e) * self.detectors[detectorName].pixelEfficiency(e)
                else:
                    angularAcceptanceEfficiency = self.detectors[detectorName].calculateIndirectAngularEfficiency()
                    conversion = self.detectors[detectorName].scintillator.emission
                    photons_detected_per_pixel_perEnergy = lambda e : photons_per_pixel_perEnergy(e) * self.detectors[detectorName].pixelEfficiency(e) * angularAcceptanceEfficiency * conversion(e)
                    
                photons_detected_per_pixel_throughSample_perEnergy = lambda e, thickness : photons_detected_per_pixel_perEnergy(e) * self.samples[sampleName].transmission(e, thickness)
                
                # The spectrum has been converted into a function of energy
                # then a flux density vs energy, and now a photons detected
                # per pixel vs energy, numerically integrate over the supplied
                # energy range
                #if self.source.energy_range is None:
                #    energy_range = self.sample[sampleName].energy_range
                #else:
                #    energy_range = self.source.energy_range
                energy_range = self.energyRanges[(sampleName, detectorName)]
                resolution = 0.001
                energy_range = np.linspace(energy_range[0], energy_range[1], int((energy_range[1]-energy_range[0])/resolution)+1)#np.arange(energy_range[0], energy_range[1], 0.1)
                photons_detected_per_pixel = np.sum(photons_detected_per_pixel_perEnergy(energy_range))*resolution
                photons_detected_per_pixel_throughSample = lambda thickness : np.sum(photons_detected_per_pixel_throughSample_perEnergy(energy_range, thickness))*resolution
                #*resolution artifically scales monochromatic beam down to less flux
                ##################################################################
                # Energy needs to be integrated out in the spectrum
                # which is why spectrum is a list
                ##################################################################
                
                # Register variables
                spec = {'photons_detected_per_pixel' : photons_detected_per_pixel,
                        'photons_detected_per_pixel_throughSample' : photons_detected_per_pixel_throughSample,
                        'photons_detected_per_pixel_perEnergy' : lambda e : photons_detected_per_pixel_perEnergy(e)*resolution,
                        'photons_detected_per_pixel_perEnergy_throughSample' : lambda e, thickness : photons_detected_per_pixel_throughSample_perEnergy(e, thickness)*resolution}
            return spec
        
        elif self.microscope.microscopeType == 'CRL':
            if self.source.spotShape == 'custom':
                ##################################################################
                # Need to figure out how custom beam profiles affect things
                # I think maybe just calculate photonCounts per pixel for each
                # index of the beam profile?
                # Either that or interpolate through space, but that seems unnecessary
                ##################################################################
                pass
            else:
                fluxDensity_atImage = self.precalculations[(sampleName, detectorName)]['fluxDensity_atImage']
                pixelArea = self.detectors[detectorName].pixelSize**2
                photons_per_pixel_perEnergy = lambda e : pixelArea * fluxDensity_atImage(e)
                
                if self.detectors[detectorName].direct:
                    photons_detected_per_pixel_perEnergy = lambda e : photons_per_pixel_perEnergy(e) * self.detectors[detectorName].pixelEfficiency(e)
                else:
                    angularAcceptanceEfficiency = self.detectors[detectorName].calculateIndirectAngularEfficiency()
                    conversion = self.detectors[detectorName].scintillator.emission
                    photons_detected_per_pixel_perEnergy = lambda e : photons_per_pixel_perEnergy(e) * self.detectors[detectorName].pixelEfficiency(e) * angularAcceptanceEfficiency * conversion(e)
                    
                photons_detected_per_pixel_throughSample_perEnergy = lambda e, thickness : photons_detected_per_pixel_perEnergy(e) * self.samples[sampleName].transmission(e, thickness)
                
                # The spectrum has been converted into a function of energy
                # then a flux density vs energy, and now a photons detected
                # per pixel vs energy, numerically integrate over the supplied
                # energy range
                #if self.source.energy_range is None:
                #    energy_range = self.sample[sampleName].energy_range
                #else:
                #    energy_range = self.source.energy_range
                if self.source.spectrum_type[0] == 'Import':
                    energy_range = self.energyRanges[(sampleName, detectorName)]
                    resolution = 0.1
                    energy_range = np.linspace(energy_range[0], energy_range[1], int((energy_range[1]-energy_range[0])/resolution)+1)#np.arange(energy_range[0], energy_range[1], 0.1)
                    photons_detected_per_pixel = np.sum(photons_detected_per_pixel_perEnergy(energy_range))*resolution
                    photons_detected_per_pixel_throughSample = lambda thickness : np.sum(photons_detected_per_pixel_throughSample_perEnergy(energy_range, thickness))*resolution
                if self.source.spectrum_type[0] == 'Const':
                    energy_range = self.source.spectrum_type[1]
                    resolution = 0.1
                    energy_range = np.linspace(energy_range[0], energy_range[1], int((energy_range[1]-energy_range[0])/resolution)+1)#np.arange(energy_range[0], energy_range[1], 0.1)
                    photons_detected_per_pixel = np.sum(photons_detected_per_pixel_perEnergy(energy_range))*resolution
                    photons_detected_per_pixel_throughSample = lambda thickness : np.sum(photons_detected_per_pixel_throughSample_perEnergy(energy_range, thickness))*resolution
                if self.source.spectrum_type[0] == 'Mono':
                    energy = self.source.spectrum_type[1]
                    photons_detected_per_pixel = photons_detected_per_pixel_perEnergy(energy)
                    photons_detected_per_pixel_throughSample = lambda thickness : photons_detected_per_pixel_throughSample_perEnergy(energy, thickness)
                else:
                    assert False, "Unknown Spectrum Type"
                ##################################################################
                # Energy needs to be integrated out in the spectrum
                # which is why spectrum is a list
                ##################################################################
                
                # Register variables
                spec = {'photons_detected_per_pixel' : photons_detected_per_pixel,
                        'photons_detected_per_pixel_throughSample' : photons_detected_per_pixel_throughSample,
                        'photons_detected_per_pixel_perEnergy' : lambda e : photons_detected_per_pixel_perEnergy(e),
                        'photons_detected_per_pixel_perEnergy_throughSample' : lambda e, thickness : photons_detected_per_pixel_throughSample_perEnergy(e, thickness)}
            return spec
        
        elif self.microscope.microscopeType == 'bragg':
            if self.source.spotShape == 'custom':
                ##################################################################
                # Need to figure out how custom beam profiles affect things
                # I think maybe just calculate photonCounts per pixel for each
                # index of the beam profile?
                # Either that or interpolate through space, but that seems unnecessary
                ##################################################################
                pass
            else:
                fluxDensity_atImage = self.precalculations[(sampleName, detectorName)]['fluxDensity_atImage'] #lambda e, e_, hkl, miscut
                pixelArea = self.detectors[detectorName].pixelSize**2
                photons_per_pixel_perEnergy = lambda e, e_, hkl, miscut : pixelArea * fluxDensity_atImage(e, e_, hkl, miscut)
                
                # Note, the Bragg magnifier is a monochromator, but this has already been accounted for in the fluxDensity_atImage
                if self.detectors[detectorName].direct:
                    photons_detected_per_pixel_perEnergy = lambda e, e_, hkl, miscut : photons_per_pixel_perEnergy(e, e_, hkl, miscut) * self.detectors[detectorName].pixelEfficiency(e)
                else:
                    angularAcceptanceEfficiency = self.detectors[detectorName].calculateIndirectAngularEfficiency()
                    conversion = self.detectors[detectorName].scintillator.emission
                    photons_detected_per_pixel_perEnergy = lambda e, e_, hkl, miscut : photons_per_pixel_perEnergy(e, e_, hkl, miscut) * self.detectors[detectorName].pixelEfficiency(e) * angularAcceptanceEfficiency * conversion(e)
                    
                photons_detected_per_pixel_throughSample_perEnergy = lambda e, e_, hkl, miscut, thickness : photons_detected_per_pixel_perEnergy(e, e_, hkl, miscut) * self.samples[sampleName].transmission(e, thickness)
                
                # The spectrum has been converted into a function of energy
                # then a flux density vs energy, and now a photons detected
                # per pixel vs energy, numerically integrate over the supplied
                # energy range
                #if self.source.energy_range is None:
                #    energy_range = self.sample[sampleName].energy_range
                #else:
                #    energy_range = self.source.energy_range
                energy_range = self.energyRanges[(sampleName, detectorName)]
                energy_range = np.linspace(energy_range[0], energy_range[1], int((energy_range[1]-energy_range[0])/0.1)+1)#np.arange(energy_range[0], energy_range[1], 0.1)
                # Now, e is the energy at which the hkl is aligned
                photons_detected_per_pixel = lambda e, hkl, miscut : photons_detected_per_pixel_perEnergy(e, e, hkl, miscut) # Dirac kills sum
                photons_detected_per_pixel_throughSample = lambda e, hkl, miscut, thickness : photons_detected_per_pixel_throughSample_perEnergy(e, e, hkl, miscut, thickness)
                ##################################################################
                # Energy needs to be integrated out in the spectrum
                # which is why spectrum is a list
                ##################################################################
                
                # Register variables
                spec = {'photons_detected_per_pixel' : photons_detected_per_pixel,
                        'photons_detected_per_pixel_throughSample' : photons_detected_per_pixel_throughSample}
            return spec
        
        else:
            assert False, "Microscope Type Not Programmed"
                
        
    def calculate_resolutionMagnification(self, sampleName : str, detectorName : str):
        if self.microscope.microscopeType == 'radiograph':
            spotSize_atSample = self.precalculations[(sampleName, detectorName)]['spotSize_atSample']
            spotSize_atImage = self.precalculations[(sampleName, detectorName)]['spotSize_atImage']
            
            if self.source.spotShape == 'circle':
                xray_mag = spotSize_atImage / (spotSize_atSample + 1e-5)
            elif self.source.spotShape == 'rectangle' or self.source.spotShape == 'custom':
                xray_mag = np.mean([spotSize_atImage[0] / (spotSize_atSample[0] + 1e-5), spotSize_atImage[1] / (spotSize_atSample[1] + 1e-5)])
            else:
                assert False, "Incorrect Beam Shape"
            
            if self.detectors[detectorName].direct:
                total_mag = xray_mag
            else:
                optical_mag = self.detectors[detectorName].calculateOpticalMagnification()
                total_mag = xray_mag * optical_mag # was previously /, don't know if that was a typo, double check simulations
            
            gauge = self.detectors[detectorName].calculateGuage(total_mag)
        
            # Register variables
            spec = {'xray_magnification' : xray_mag,
                    'gauge' : gauge}
            if not self.detectors[detectorName].direct:
                spec['optical_magnification'] = optical_mag
            return spec
        
        elif self.microscope.microscopeType == 'CRL':
            xray_mag = self.microscope.magnification
            
            if self.detectors[detectorName].direct:
                total_mag = lambda e : xray_mag(e)
            else:
                optical_mag = self.detectors[detectorName].calculateOpticalMagnification()
                total_mag = lambda e : xray_mag(e) * optical_mag # was previously /, don't know if that was a typo, double check simulations
            
            gauge = lambda e : self.detectors[detectorName].calculateGuage(total_mag(e))
        
            # Register variables
            spec = {'xray_magnification' : xray_mag,
                    'gauge' : gauge}
            if not self.detectors[detectorName].direct:
                spec['optical_magnification'] = optical_mag
            return spec
        
        elif self.microscope.microscopeType == 'bragg':
            if self.microscope.N_bounce == 1 or self.microscope.N_bounce == 2:
                xray_mag = self.microscope.totalXMag #(e, hkl, miscut)
            elif self.microscope.N_bounce == 4:
                xray_mag = self.microscope.totalXMag_1D #(e, hkl, miscuts)
            else:
                assert False, "Incorrect number of bounces in bragg magnifier"
            
            if self.detectors[detectorName].direct:
                total_mag = lambda e, hkl, miscut : xray_mag(e, hkl, miscut)
            else:
                optical_mag = self.detectors[detectorName].calculateOpticalMagnification()
                total_mag = lambda e, hkl, miscut : xray_mag(e, hkl, miscut) * optical_mag
            
            gauge = lambda e, hkl, miscut : self.detectors[detectorName].calculateGuage(total_mag(e, hkl, miscut))
        
            # Register variables
            spec = {'xray_magnification' : xray_mag,
                    'gauge' : gauge}
            if not self.detectors[detectorName].direct:
                spec['optical_magnification'] = optical_mag
            return spec
        
        else:
            assert False, "Microscope Type Not Programmed"
            
    
    def calculate_beamCameraOverlap(self, sampleName : str, detectorName : str):
        if self.microscope.microscopeType == 'radiograph':
            if self.source.spotShape == 'circle':
                db = self.precalculations[(sampleName, detectorName)]['spotSize_atImage']
                
                if not self.detectors[detectorName].direct:
                    db *= self.detectors[detectorName].calculateOpticalMagnification()
                
                hc, wc = self.detectors[detectorName].pixelCount
                beam_captured_by_camera, camera_filled_by_beam = circle_overlap_onto_rectangle(db, hc, wc)
                '''
                if db <= np.min([hc, wc]):
                    beam_captured_by_camera = 1.0
                    camera_filled_by_beam = (np.pi*(db/2)**2) / (hc*wc + 1e-25)
                elif db >= np.sqrt(hc**2 + wc**2):
                    camera_filled_by_beam = 1.0
                    beam_captured_by_camera = (hc*wc) / (np.pi*(db/2)**2 + 1e-25)
                elif db >= np.min([hc, wc]) and db <= np.max([hc, wc]):
                    h = (db - np.min(hc, wc)) / 2
                    r = db/2
                    diff = r**2 * np.arccos(1 - h/r) - (r-h)*np.sqrt(r**2 - (r-h)**2)
                    diff *= 2
                    area = np.pi * r**2
                    beam_captured_by_camera = (area - diff) / (area + 1e-25)
                    camera_filled_by_beam = (area - diff) / (hc*wc + 1e-25)
                else:
                    h1 = (db - hc) / 2
                    h2 = (db - wc) / 2
                    r = db/2
                    diff1 = r**2 * np.arccos(1 - h1/r) - (r-h1)*np.sqrt(r**2 - (r-h1)**2)
                    diff1 *= 2
                    diff2 = r**2 * np.arccos(1 - h2/r) - (r-h2)*np.sqrt(r**2 - (r-h2)**2)
                    diff2 *= 2
                    area = np.pi * r**2
                    beam_captured_by_camera = (area - diff1 - diff2) / (area + 1e-25)
                    camera_filled_by_beam = (area - diff1 - diff2) / (hc*wc + 1e-25)
                '''
                    
            elif self.source.spotShape == 'rectangle' or self.source.spotShape == 'custom':
                hb, wb = self.precalculations[(sampleName, detectorName)]['spotSize_atImage']
                hc, wc = self.detectors[detectorName].pixelCount
                
                if not self.detectors[detectorName].direct:
                    hb *= self.detectors[detectorName].calculateOpticalMagnification()
                    wb *= self.detectors[detectorName].calculateOpticalMagnification()
                
                '''
                beam_captured_by_camera = np.min([hb, hc]) * np.min([wb, wc]) / (hb*wb + 1e-25)
                camera_filled_by_beam = np.min([hb, hc]) * np.min([wb, wc]) / (hc*wc + 1e-25)
                '''
                beam_captured_by_camera, camera_filled_by_beam = rectangle_overlap_onto_rectangle(hb, wb, hc, wc)
            else:
                assert False, "Incorrect Spot Shape"
            
            spec = {'beam_captured_by_camera' : beam_captured_by_camera,
                    'camera_filled_by_beam' : camera_filled_by_beam}
            return spec
        
        elif self.microscope.microscopeType == 'CRL':
            if self.source.spotShape == 'circle':
                db_ = lambda e : self.precalculations[(sampleName, detectorName)]['spotSize_atImage'](e)
                
                if not self.detectors[detectorName].direct:
                    db = lambda e : db_(e)*self.detectors[detectorName].calculateOpticalMagnification()
                else:
                    db = db_ # Needed to prevent recursive issues in above line.
                
                hc, wc = self.detectors[detectorName].pixelCount
                beam_captured_by_camera_AND_camera_filled_by_beam = lambda e : circle_overlap_onto_rectangle(db(e), hc, wc)
                    
            elif self.source.spotShape == 'rectangle' or self.source.spotShape == 'custom':
                hb_, wb_ = self.precalculations[(sampleName, detectorName)]['spotSize_atImage']
                hc, wc = self.detectors[detectorName].pixelCount
                
                if not self.detectors[detectorName].direct:
                    hb = lambda e : hb_(e)*self.detectors[detectorName].calculateOpticalMagnification()
                    wb = lambda e : wb_(e)*self.detectors[detectorName].calculateOpticalMagnification()
                else:
                    hb = hb_
                    wb = wb_
                    
                beam_captured_by_camera_AND_camera_filled_by_beam = lambda e : rectangle_overlap_onto_rectangle(hb(e), wb(e), hc, wc)
            else:
                assert False, "Incorrect Spot Shape"
            
            spec = {'beam_captured_by_camera' : lambda e : beam_captured_by_camera_AND_camera_filled_by_beam(e)[0],
                    'camera_filled_by_beam' : lambda e : beam_captured_by_camera_AND_camera_filled_by_beam(e)[1]}
            return spec
        
        elif self.microscope.microscopeType == 'bragg':
            if self.source.spotShape == 'circle':
                db_ = self.precalculations[(sampleName, detectorName)]['spotSize_atImage'] #(e, hkl, miscut)
                
                if not self.detectors[detectorName].direct:
                    if callable(db_):
                        db = lambda e, hkl, miscut : db_(e, hkl, miscut)*self.detectors[detectorName].calculateOpticalMagnification()
                    else:
                        db = lambda e, hkl, miscut : db_*self.detectors[detectorName].calculateOpticalMagnification()
                else:
                    db = db_ # Needed to prevent recursive issues in above line.
                
                hc, wc = self.detectors[detectorName].pixelCount
                beam_captured_by_camera_AND_camera_filled_by_beam = lambda e, hkl, miscut : circle_overlap_onto_rectangle(db(e, hkl, miscut), hc, wc)
                    
            elif self.source.spotShape == 'rectangle' or self.source.spotShape == 'custom':
                hb_, wb_ = self.precalculations[(sampleName, detectorName)]['spotSize_atImage']
                hc, wc = self.detectors[detectorName].pixelCount
                
                if not self.detectors[detectorName].direct:
                    if callable(hb_):
                        hb = lambda e, hkl, miscut : hb_(e, hkl, miscut)*self.detectors[detectorName].calculateOpticalMagnification()
                    else:
                        hb = lambda e, hkl, miscut : hb_*self.detectors[detectorName].calculateOpticalMagnification()
                    if callable(wb_):
                        wb = lambda e, hkl, miscut : wb_(e, hkl, miscut)*self.detectors[detectorName].calculateOpticalMagnification()
                    else:
                        wb = lambda e, hkl, miscut : wb_*self.detectors[detectorName].calculateOpticalMagnification()
                else:
                    hb = hb_
                    wb = wb_
                    
                beam_captured_by_camera_AND_camera_filled_by_beam = lambda e, hkl, miscut : rectangle_overlap_onto_rectangle(hb(e, hkl, miscut), wb(e, hkl, miscut), hc, wc)
            else:
                assert False, "Incorrect Spot Shape"
            
            spec = {'beam_captured_by_camera' : lambda e, hkl, miscut : beam_captured_by_camera_AND_camera_filled_by_beam(e, hkl, miscut)[0],
                    'camera_filled_by_beam' : lambda e, hkl, miscut : beam_captured_by_camera_AND_camera_filled_by_beam(e, hkl, miscut)[1]}
            return spec
        
        else:
            assert False, "Microscope Type Not Programmed"
    
    ##########################################################################
    # Remember for indirect detection to ensure the scintillator doesn't
    # clip the beam.
    ##########################################################################
    
    ##########################################################################
    # beam-captured_by_camera and photons_detected_per_pixel seem to not be
    # consistent? Check and register angular efficiency from NA
    ##########################################################################
    
    ##########################################################################
    # Source divergence does not seems to significantly change spot size,
    # which is not correct.
    ##########################################################################
    
    ##########################################################################
    # Will need to go back through and figure out how to make these properties
    # of arbitrary subsets of the inputs
    # and how to optimize as desired
    ##########################################################################
