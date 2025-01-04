# X-ray_Photometrics

These packages are meant to provide the tools necessary to estimate the performance of X-ray microscopes for radiography and transmission X-ray microscopy (TXM). They provide a Python implementation for two primary applications:
* Calculating the magnification and imaging geometry for X-ray microscopes using compound refractive lenses (CRL), based on Simons et al. (2017): doi:10.1107/S160057751602049X
* Transmission through complex media based on data from the CXRO and NIST databases: https://henke.lbl.gov/optical_constants/ and https://physics.nist.gov/PhysRefData/FFast/html/form.html

Note, some basic functionality is included to accomodate calculating the magnification of Bragg-magnifier based microscopes based on Vagovic et al. (2011): doi:10.1107/S090904951102989X

For both of these applications, the calculations are built on ray-optical assumptions with purely attenuation-based contrast mechanisms. Beer-Lambert's law for a continuously varying medium is given by

$$
\mathcal{T}(E) = \prod \big(e^{-1/\mu}\big)^{dz} = \exp\int-\frac{1}{\mu}dz = \exp\sum_i \frac{-t_i}{\mu_i}
$$

where $T$ is the transmission as a function of X-ray photon energy $E$, and $\mu$ is the attenuation length. Note, $dz$ is the differential element along the transmission axis ($z$) and $t_i$ is the thickness of material $i$. Several equivalent notations have been introduced in the above equation, but the last is the most common and relevant to discretely varied materials.

## Presumed Ray-Optical Image Formation Model

From this, a calculation based on flux densities can be used to estimate the count-rate on a detector (neglecting noise) is

$$
I = \int \Gamma\cdot\mathcal{C}\mathcal{A}\mathcal{T} dE
$$

for X-ray source power spectrum $\Gamma$, detector conversion efficiency $\mathcal{C}$, detector absorption $\mathcal{A}$, and sample transmission $\mathcal{T}$.

* The source power spectrum, $\Gamma$, is usually measured but defined by the Bremsstrahlung for a lab-based X-ray source, given by the beamline scientists at a synchrotron, or approximated as a Guassian or Dirac delta (depending on application sensitivity to chromaticity) at an X-ray Free Electron Laser (XFEL).
* The detector conversion efficiency, $\mathcal{C}$, varies whether the detector is direct or indirect but captures the flux density changes due to magnification. For a direct detector,

$$
\mathcal{C}=\frac{\Delta x^2}{M^2}C_{ISO}
$$

assuming isotropic 2D effective magnification $M$ and square pixels of size $\Delta x$ with digital gain $C_{ISO}$. For an indirect detector, this can be written as

$$
\mathcal{C}=\frac{\Delta x^2}{M^2}C\cdot\frac{1}{M_o^2}C_{ISO}QE
$$

where the first factor is flux loss through the X-ray microscope including the scintillator conversion efficiency $C$, and the second factor is the optical efficiency with optical magnification $M_o$, digital gain $C_{ISO}$, and visible detector quantum efficiency $QE$.
* The detector absorption is given by $\mathcal{A} = 1 - e^{-t_d/\mu_d}$ for scintillator (indirect) and direct detectors where $t_d$ is the thickness of the X-ray absorbing element and $\mu_d$ is the attenuation length of that absorbing element. For a direct detector, $\mathcal{A}$ would often be considered the quantum efficiency (QE) of the detector, while for indirect, scintillator-based detectors, it is the absorbtion of the scintillator which goes into fluorescence detected by visible detectors.
* The sample transmission function, $\mathcal{T}$, is given by the Beer-Lambert law described above, based on data from the CXRO and NIST databases.

This describes a very simple image formation model for purely attenuation-based contrast mechanisms, which is assumed as the basis of this work and valid for large features in near-field radiography or TXM in the imaging condition, for which phase-contrast is minimized.

## Implementation of Radiography

In radiography, the X-ray magnification arises from the X-ray divergence and sample-to-detector distance.

## Implementation of TXM using CRLs

As mentioned previously, the TXM assumed in this work is based on compound refractive lenses (CRLs) and uses the results of the ray-transfer matrix formalism described in Simons et al. (2017).

In Simons et al., all the microscope parameters of interest such as the magnification are functions of the microscope geometry and the microscope geometry, point spread function (PSF), etc. are in-turn functionally dependent on the magnification. Since neither the microscope geometry nor the magnification are known *a priori*, this code uses numerical root-finding methods to solve for $M$ in Eq. 21 of Simons et al. Once the magnification is calculated from the availabile microscope length and CRL properties, the microscope geometry and imaging condition (sample-to-lens and lens-to-detector distances) are calculated using Eq. 18 and 19 of Simons et al.

Valid CRL materials are beryllium, diamond, aluminum, silicon and germanium, though CRLs are rarely if ever made from silicon or germanium.

## Implementation of Bragg Magnifiers

The magnification resulting from a Bragg magnifier, which uses diffraction from an asymmetrically cut crystal to expand the beam, is described in Vagovic et al. (2011) as well as many other works.

The term for magnification is based on Eq. 1 of Vagovic et al. where the refractive corrections $\delta_{i,h}^{1,2}$ are neglected and most important for very glancing angles. The total external reflection angle is calculated and can be used to place bounds on the maximal magnification. The acceptance angle / bandwidth of the diffractive crystal is also calculable, but currently not integrated into estimating fluence from the source spectrum.

Valid Bragg magnifier crystals in this code are beryllium, diamond, aluminum, silicon and germanium, though Bragg magnifiers are almost always made from silicon or germanium.

## References

[1] Simons 2017, "Simulating and optimizing compound refractive lens-based X-ray microscopes" doi:10.1107/S160057751602049X

[2] Henke 1993, "X-ray interactions: photoabsorption, scattering, transmission, and reflection at E=50-30000 eV, Z=1-92"
  - CXRO: https://henke.lbl.gov/optical_constants/

[3] Chantler 1997, "Detailed Tabulation of Atomic Form Factors, Photoelectric Absorption and Scattering Cross Section, and Mass Attenuation Coefficients for Z = 1-92 from E = 1-10 eV to E = 0.4-1.0 MeV" doi:10.18434/T4HS32
  - NIST: https://physics.nist.gov/PhysRefData/FFast/html/form.html

[4] Vagovic 2011, "In-line Bragg magnifier based on V-shaped germanium crystals" doi:10.1107/S090904951102989X

[5] Spal 2001, "Submicron resolution hard x-ray holography with asymmetric bragg diffraction microscope" doi:10.1103/PhysRevLett.86.3044

