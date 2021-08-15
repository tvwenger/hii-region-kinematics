"""
kinematic_model.py
Generate a synthetic observation of a model HII region. Allow for
different and multiple internal kinematics of the nebulae:
- stationary
- solid body rotation
- differential rotation
- bipolar outflows
- expansion

Copyright(C) 2021 by
Trey V. Wenger; tvwenger@gmail.com

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Trey Wenger & Dana Balser - v2.0 - August 2021
Standardized for package release
"""

import astropy.units as u

from .hii_region import HIIRegion


def main(
    electron_density,
    electron_temp,
    diameter,
    kinematics=[],
    grid_size=128,
    num_channels=256,
    vel_width=200.0,
    rrl_n=85,
    delta_n=1,
    nonthermal_fwhm=15.0,
    distance=5.0,
    beam_fwhm=90.0,
    image_size=600.0,
    noise=0.001,
    velocity_fits="velocity.fits",
    sky_brightness_fits="sky_brightness.fits",
    obs_brightness_fits="obs_brightness.fits",
    center_fits="center.fits",
    e_center_fits="e_center.fits",
    fwhm_fits="fwhm.fits",
    e_fwhm_fits="e_fwhm.fits",
):
    """
    Generate an HII region model and a synthetic observation, save
    data to FITS images.

    Inputs:
      electron_density :: scalar (cm-3)
        Nebular electron density
      electron_temp :: scalar (K)
        Nebular electron temperature
      diameter :: scalar (pc)
        Nebula diameter
      kinematics :: List of dictionaries
        Internal nebula kinematics. See hii_region.HIIRegion
      grid_size :: integer
        Number of grid points along side of model (must be even)
      num_channels :: integer
        Number of velocity channels
      vel_width :: scalar (km/s)
        Width of velocity range (centered on 0 km/s)
      rrl_n :: integer
        RRL principal quantum number
      delta_n :: integer
        RRL principal quantum number transition
      nonthermal_fwhm :: scalar (km/s)
        RRL non-thermal FWHM width
      distance :: scalar (kpc)
        Distance of nebula
      beam_fwhm :: scalar (arcsec)
        FWHM beam width
      image_size :: scalar (arcsec)
        Length of image side
      noise :: scalar (mJy/arcsec2)
        Pre-convolution sky brightness noise
      velocity_fits :: string
        FITS image where model velocity grid is saved
      sky_brightness_fits :: string
        FITS image where true sky brightness is saved
      obs_brightness_fits :: string
        FITS image where observed sky brightness is saved
      center_fits, e_center_fits :: strings
        FITS images where fitted Gaussian center velocity is saved
      fwhm_fits, e_fwhm_fits :: strings
        Fits images where fitted Gaussian FWHM is saved

    Returns: Nothing
    """
    if grid_size % 2 != 0:
        raise ValueError("grid_size must be even integer")

    # HII region morphology and physical properties
    electron_density = electron_density * u.cm ** -3.0
    electron_temp = electron_temp * u.K
    diameter = diameter * u.pc
    model = HIIRegion(
        electron_density=electron_density,
        electron_temp=electron_temp,
        diameter=diameter,
        kinematics=kinematics,
    )

    # Grid parameters
    vel_width = vel_width * u.km / u.s
    nonthermal_fwhm = nonthermal_fwhm * u.km / u.s
    model.set_grid(
        grid_size, num_channels, vel_width=vel_width, velocity_fitsfile=velocity_fits
    )
    model.set_rrl(rrl_n, nonthermal_fwhm, delta_n=delta_n)
    model.calc_brightness_grid(fitsfile=sky_brightness_fits)

    # Observation parameters
    distance = distance * u.kpc
    beam_fwhm = beam_fwhm * u.arcsec
    image_size = image_size * u.arcsec
    noise = noise * u.mJy / u.arcsec ** 2.0
    model.observe(
        distance, beam_fwhm, image_size, noise=noise, fitsfile=obs_brightness_fits
    )

    # Fit Gaussians, generate maps of VLSR and FWHM
    model.fit(
        center_fitsfile=center_fits,
        e_center_fitsfile=e_center_fits,
        fwhm_fitsfile=fwhm_fits,
        e_fwhm_fitsfile=e_fwhm_fits,
    )
