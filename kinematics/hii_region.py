"""
hii_region.py
Class definition for HII region model

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

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

import astropy.units as u
import astropy.constants as c
from astropy.io import fits

from . import physics


def gaussian(x, amp, center, fwhm, background):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return amp * np.exp(-0.5 * ((x - center) / sigma) ** 2.0) + background


class HIIRegion:
    """
    HII region class defines the nebular morphology and physical
    properties, and has methods to apply radiative transfer to
    determine the sky brightness distribution.
    """

    def __init__(
        self,
        electron_density=50.0 * u.cm ** -3,
        electron_temp=8000.0 * u.K,
        diameter=10.0 * u.pc,
        kinematics=[],
    ):
        """
        Initialize a new HIIRegion object. Nebula is assumed
        homogenous and isothermal.

        Inputs:
          electron_density :: scalar (with units)
            Electron density
          electron_temp :: scalar (with units)
            Electron temperature
          diameter :: scalar (with units)
            HII region diameter
          kinematics :: list of dictionaries
            Kinematic components of nebula. Available values are one
            or more of (all values require units):

            {'type': 'solidbody',
             'eq_speed': <equatorial rotation speed>,
             'sky_pa': <plane-of-sky position angle>,
             'los_pa': <line-of-sight position angle>}
            For solid body rotation, the rotation speed is given by:
            v(R, phi) = angular_speed*R*cos^2(phi),
            where phi is the latitudinal angle. The position angles
            define the orientation of angular momentum vector and the
            nebula's equator.

            {'type': 'differential',
             'eq_speed': <equatorial rotation speed at surface>,
             'r_power': <exponent of radial change in rotation speed>,
             'sky_pa': <plane-of-sky position angle>,
             'los_pa': <line-of-sight position angle>}
            For differential rotation, the rotation speed is given by:
            v(R, phi) = eq_angular_speed*(R/radius)^r_power*cos^2(phi)
            where radius is the radius of the nebula and phi is the
            latitudinal angle. The position angles define the
            orientation of angular momentum vector and the nebula's
            equator.

            {'type': 'outflow',
             'speed': <outflow speed>,
             'angle': <opening angle>,
             'sky_pa': <plane-of-sky position angle>,
             'los_pa': <line-of-sight position angle>}
            For a bipolar outflow, the velocity of gas within the
            outflows is radial with a speed given by:
            v(R, phi) = {speed : if phi < angle/2.0,
                         0     : otherwise}
            where phi is the angle from the polar axis. The position
            angle defines the postive radial velocity direction of
            the outflow axis.

            {'type': 'expansion',
             'alpha': <expansion speed at surface>,
             'beta': <power law exponent>}
            For an expanding nebula, the spherically symmetric expansion
            velocity is given by
            v(R) = alpha * (R/radius)^beta

        Returns: hii_region
          hii_region :: HIIRegion object
            New HIIRegion object
        """
        self.electron_density = electron_density
        self.electron_temp = electron_temp
        self.diameter = diameter
        self.kinematics = kinematics
        self.velocity_axis = self.frequency_axis = None
        self.chan_size = self.cell_size = None
        self.x_axis = self.y_axis = self.z_axis = None
        self.electron_density_grid = self.electron_temp_grid = None
        self.velocity_grid = self.center_frequency_grid = None
        self.rrl_n = self.nonthermal_fwhm = self.rest_freq = None
        self.thermal_fwhm_grid = self.fwhm_grid = None
        self.distance = self.image_size = self.obs_pixel_size = None
        self.brightness_grid = self.obs_brightness_grid = None

    def set_grid(
        self,
        n_positions,
        n_channels,
        vel_width=200.0 * u.km / u.s,
        velocity_fitsfile="velocity.fits",
    ):
        """
        Define the model position-velocity grid. The position grid
        has width equal to the nebula diameter + 10%, and the velocity
        axis has defined width.

        Inputs:
          n_positions :: integer
            Number of cells along each position dimension
          n_channels :: integer
            Number of velocity bins
          vel_width :: scalar (with units)
            Width of velocity axis
          velocity_fitsfile :: string
            Filename of saved velocity FITS image

        Returns: Nothing
        """
        self.n_grid = n_positions
        self.n_chan = n_channels
        self.vel_width = vel_width
        vel_width = vel_width.to("km/s").value
        chan_size = vel_width / (n_channels - 1.0)
        self.velocity_axis = (
            np.arange(-vel_width / 2.0, vel_width / 2.0 + chan_size, chan_size)
            * u.km
            / u.s
        )
        self.chan_size = chan_size * u.km / u.s
        grid_size = 1.1 * self.diameter.to("pc").value
        cell_size = grid_size / (n_positions - 1.0)
        self.x_axis = np.linspace(-grid_size / 2.0, grid_size / 2.0, n_positions) * u.pc
        self.y_axis = np.linspace(-grid_size / 2.0, grid_size / 2.0, n_positions) * u.pc
        self.z_axis = np.linspace(-grid_size / 2.0, grid_size / 2.0, n_positions) * u.pc
        self.cell_size = cell_size * u.pc

        # Convention: x increases to right(decreasing RA), y increases
        # with distance, and z increases up (increasing Dec.)
        # azimuth defined as zero along +x, increasing toward +y
        # elevation defined as 0 on z=0 plane, increasing toward +z
        # sky PA defined as 0 along +z, increasing toward -x
        # los PA defined as 0 on y=0 plane, increasing toward +y
        sky_vector = (
            np.array(
                np.meshgrid(
                    self.x_axis.to("pc").value,
                    self.y_axis.to("pc").value,
                    self.z_axis.to("pc").value,
                    indexing="ij",
                )
            )
            * u.pc
        )
        R_grid = np.sqrt(np.sum(sky_vector ** 2.0, axis=0))

        # Initialize nebula density and temperature grids
        interior = R_grid < self.diameter / 2.0
        self.electron_density_grid = np.zeros(R_grid.shape) * u.cm ** -3.0
        self.electron_density_grid[interior] = self.electron_density
        self.electron_temp_grid = np.zeros(R_grid.shape) * u.K
        self.electron_temp_grid[interior] = self.electron_temp

        # Initialize nebula velocity grid
        self.velocity_grid = np.zeros(R_grid.shape) * u.km / u.s
        for kinematic in self.kinematics:
            # Rotate nebula clockwise los_pa about x-axis,
            # then counter-clockwise sky_pa about y-axis
            rot_x = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, np.cos(kinematic["los_pa"]), -np.sin(kinematic["los_pa"])],
                    [0.0, np.sin(kinematic["los_pa"]), np.cos(kinematic["los_pa"])],
                ]
            )
            rot_y = np.array(
                [
                    [np.cos(kinematic["sky_pa"]), 0.0, np.sin(kinematic["sky_pa"])],
                    [0.0, 1.0, 0.0],
                    [-np.sin(kinematic["sky_pa"]), 0.0, np.cos(kinematic["sky_pa"])],
                ]
            )
            rot = np.dot(rot_x, rot_y)
            rot_vector = (
                np.tensordot(rot, sky_vector.to("pc").value, axes=(1, 0)) * u.pc
            )
            rot_R = np.sqrt(np.sum(rot_vector ** 2.0, axis=0))
            rot_azimuth = np.arctan2(rot_vector[1], rot_vector[0])
            rot_elevation = np.arcsin(rot_vector[2] / rot_R)

            # Solid body rotation with angular momentum vector
            # along rotated +z axis
            if kinematic["type"] == "solidbody":
                speed = rot_R * kinematic["eq_speed"] / (self.diameter / 2.0)
                speed *= np.cos(rot_elevation) ** 2.0
                speed = speed.to("km/s").value
                rot_velocity = (
                    np.array(
                        [
                            -speed * np.sin(rot_azimuth),
                            speed * np.cos(rot_azimuth),
                            np.zeros(speed.shape),
                        ]
                    )
                    * u.km
                    / u.s
                )

            # Differential rotation with angular momentum vector
            # along rotated +z axis
            elif kinematic["type"] == "differential":
                speed = (
                    kinematic["eq_speed"]
                    * (rot_R / (self.diameter / 2.0)) ** kinematic["r_power"]
                )
                speed *= np.cos(rot_elevation) ** 2.0
                speed = speed.to("km/s").value
                rot_velocity = (
                    np.array(
                        [
                            -speed * np.sin(rot_azimuth),
                            speed * np.cos(rot_azimuth),
                            np.zeros(speed.shape),
                        ]
                    )
                    * u.km
                    / u.s
                )

            # Bipolar outflow along rotated z-axis
            elif kinematic["type"] == "outflow":
                # get polar angle, between 0 and 90 degrees
                rot_polar = 90.0 * u.deg - rot_elevation
                rot_polar[rot_polar > 90.0 * u.deg] = (
                    180.0 * u.deg - rot_polar[rot_polar > 90.0 * u.deg]
                )
                v_radial = np.zeros(rot_R.shape) * u.km / u.s
                v_radial[rot_polar < kinematic["angle"] / 2.0] = kinematic["speed"]
                v_radial = v_radial.to("km/s").value
                rot_velocity = (
                    np.array(
                        [
                            v_radial * np.cos(rot_elevation) * np.cos(rot_azimuth),
                            v_radial * np.cos(rot_elevation) * np.sin(rot_azimuth),
                            v_radial * np.sin(rot_elevation),
                        ]
                    )
                    * u.km
                    / u.s
                )

            # Radial expansion like Whalen et al. (2018) profile
            elif kinematic["type"] == "expansion":
                v_radial = (
                    kinematic["alpha"]
                    * (rot_R / (self.diameter / 2.0)) ** kinematic["beta"]
                )
                rot_velocity = (
                    np.array(
                        [
                            v_radial * np.cos(rot_elevation) * np.cos(rot_azimuth),
                            v_radial * np.cos(rot_elevation) * np.sin(rot_azimuth),
                            v_radial * np.sin(rot_elevation),
                        ]
                    )
                    * u.km
                    / u.s
                )
            else:
                raise ValueError("Invalid kinematic type: {0}".format(kinematic))

            # Rotate back to sky frame
            sky_velocity = (
                np.tensordot(rot.T, rot_velocity.to("km/s").value, axes=(1, 0))
                * u.km
                / u.s
            )

            # Add radial (+y) component to velocity_grid
            self.velocity_grid += sky_velocity[1]

        # Save to FITS image
        # N.B. Need to transpose data to match FITS orientation
        # mask region outside of nebula
        velocity_grid = self.velocity_grid.to("km/s").value
        velocity_grid[~interior] = np.nan
        hdu = fits.PrimaryHDU(velocity_grid.T)
        hdu.header["OBJECT"] = "Model"
        hdu.header["BTYPE"] = "Velocity"
        hdu.header["BUNIT"] = "km/s"
        hdu.header["BSCALE"] = 1.0
        hdu.header["BZERO"] = 0.0
        hdu.header["EQUINOX"] = 2.0e3
        hdu.header["RADESYS"] = "FK5"
        hdu.header["CTYPE1"] = "X"
        hdu.header["CRVAL1"] = self.x_axis[0].to("pc").value
        hdu.header["CDELT1"] = self.cell_size.to("pc").value
        hdu.header["CRPIX1"] = 1
        hdu.header["CUNIT1"] = "parsec"
        hdu.header["CTYPE2"] = "Y"
        hdu.header["CRVAL2"] = self.y_axis[0].to("pc").value
        hdu.header["CDELT2"] = self.cell_size.to("pc").value
        hdu.header["CRPIX2"] = 1
        hdu.header["CUNIT2"] = "parsec"
        hdu.header["CTYPE3"] = "Z"
        hdu.header["CRVAL3"] = self.z_axis[0].to("pc").value
        hdu.header["CDELT3"] = self.cell_size.to("pc").value
        hdu.header["CRPIX3"] = 1
        hdu.header["CUNIT3"] = "parsec"
        hdu.writeto(velocity_fitsfile, overwrite=True)

    def set_rrl(self, rrl_n, nonthermal_fwhm, delta_n=1):
        """
        Define the observed RRL transition.

        Inputs:
          rrl_n :: integer
            RRL principal quantum number
          nonthermal_fwhm :: scalar (with units)
            Non-thermal FWHM line width in velocity units
          delta_n :: integer
            Principal quantum number transition

        Returns: Nothing
        """
        self.nt_fwhm_vel = nonthermal_fwhm
        self.rrl_n = rrl_n
        self.rrl_dn = delta_n
        self.rest_freq = physics.rydberg(rrl_n, delta_n=delta_n)
        # convert non-thermal fwhm to frequency units
        self.nonthermal_fwhm = self.rest_freq * nonthermal_fwhm / c.c
        self.frequency_axis = physics.doppler(self.rest_freq, self.velocity_axis)
        self.center_frequency_grid = physics.doppler(self.rest_freq, self.velocity_grid)
        self.thermal_fwhm_grid = physics.calc_thermalwidth(
            self.center_frequency_grid, self.electron_temp_grid
        )
        self.fwhm_grid = np.sqrt(
            self.thermal_fwhm_grid ** 2.0 + self.nonthermal_fwhm ** 2.0
        )

    def calc_brightness_grid(self, fitsfile="sky_brightness.fits"):
        """
        Calculate the brightness distribution (flux density per
        angular area). Save the results to a FITS image.

        Inputs:
          fitsfile :: string
            Filename of saved FITS image

        Returns: Nothing
        """
        self.brightness_grid = (
            np.zeros((self.x_axis.size, self.z_axis.size, self.velocity_axis.size))
            * u.mJy
            / u.arcsec ** 2
        )
        #
        # Too memory-intensive to compute full (position, position,
        # position, frequency) grid, so we loop over frequency
        #
        for i, freq in enumerate(self.frequency_axis):
            print("Modeling channel {0}".format(i), end="\r")
            #
            # Get free-free and RRL absorption coefficients
            #
            ffabscoeff_grid = physics.calc_ffabscoeff(
                freq, self.electron_temp_grid, self.electron_density_grid
            )
            rrlabscoeff_grid = physics.calc_rrlabscoeff(
                freq,
                self.rest_freq,
                self.center_frequency_grid,
                self.rrl_n,
                self.fwhm_grid,
                self.electron_temp_grid,
                self.electron_density_grid,
            )
            #
            # Get total (free-free + RRL) optical depth, sum along
            # y axis
            #
            optical_depth_grid = physics.calc_optical_depth(
                ffabscoeff_grid + rrlabscoeff_grid, self.cell_size, axis=1
            )
            #
            # Get brightness
            #
            self.brightness_grid[:, :, i] = physics.calc_brightness(
                freq, optical_depth_grid, self.electron_temp
            )

        # Save to FITS image
        # N.B. Need to transpose data to match FITS orientation
        hdu = fits.PrimaryHDU(self.brightness_grid.to("mJy/arcsec2").value.T)
        hdu.header["OBJECT"] = "Model"
        hdu.header["BTYPE"] = "SpectralBrightness"
        hdu.header["BUNIT"] = "mJy/arcsec2"
        hdu.header["BSCALE"] = 1.0
        hdu.header["BZERO"] = 0.0
        hdu.header["EQUINOX"] = 2.0e3
        hdu.header["RADESYS"] = "FK5"
        hdu.header["CTYPE1"] = "X"
        hdu.header["CRVAL1"] = self.x_axis[0].to("pc").value
        hdu.header["CDELT1"] = self.cell_size.to("pc").value
        hdu.header["CRPIX1"] = 1
        hdu.header["CUNIT1"] = "parsec"
        hdu.header["CTYPE2"] = "Y"
        hdu.header["CRVAL2"] = self.y_axis[0].to("pc").value
        hdu.header["CDELT2"] = self.cell_size.to("pc").value
        hdu.header["CRPIX2"] = 1
        hdu.header["CUNIT2"] = "parsec"
        hdu.header["CTYPE3"] = "VELO-LSR"
        hdu.header["CRVAL3"] = self.velocity_axis[0].to("km/s").value
        hdu.header["CDELT3"] = self.chan_size.to("km/s").value
        hdu.header["CRPIX3"] = 1
        hdu.header["CUNIT3"] = "km/s"
        hdu.header["RESTFRQ"] = self.rest_freq.to("Hz").value
        hdu.header["HISTORY"] = "Parameters of Model HII Region:"
        hdu.header["HISTORY"] = "electron density = " + str(self.electron_density)
        hdu.header["HISTORY"] = "electron temperature = " + str(self.electron_temp)
        hdu.header["HISTORY"] = "diameter = " + str(self.diameter)
        hdu.header["HISTORY"] = "distance = " + str(self.distance)
        hdu.header["HISTORY"] = "nonthermal FWHM line width = " + str(self.nt_fwhm_vel)
        hdu.header["HISTORY"] = "beam FWHM width = None"
        hdu.header["HISTORY"] = "noise = None"
        hdu.header["HISTORY"] = "number of pixels in 3D grid = " + str(self.n_grid)
        hdu.header["HISTORY"] = "number of spectral channels = " + str(self.n_chan)
        hdu.header["HISTORY"] = "velocity range = " + str(self.vel_width)
        hdu.header["HISTORY"] = "image size = " + str(self.image_size)
        hdu.header["HISTORY"] = "rrl n = " + str(self.rrl_n)
        hdu.header["HISTORY"] = "rrl dn = " + str(self.rrl_dn)
        hdu.header["HISTORY"] = "kinematic model = " + str(self.kinematics)
        hdu.writeto(fitsfile, overwrite=True)

    def observe(
        self,
        distance,
        beam_fwhm,
        image_size,
        noise=1.0 * u.mJy / u.arcsec ** 2.0,
        fitsfile="obs_brightness.fits",
    ):
        """
        Re-grid sky brightness to coarser resolution, add some noise,
        convolve with a beam, and save observed sky brightness
        distribution to a FITS image.

        Inputs:
          distance :: scalar (with units)
            Distance of nebula
          beam_fwhm :: scalar (with units)
            Angular beam FWHM
          image_size :: scalar (with units)
            Angular image size
          noise :: scalar (with units)
            Sky brightness noise
          fitsfile :: string
            Filename of saved FITS image

        Returns: Nothing
        """
        # Observed image has 10 pixels across FWHM of beam, but
        # the actual number is whatever allows integer rounding
        # of obs_pixel/model_pixel.
        self.noise = noise
        self.distance = distance
        model_pixel_size = u.rad * self.cell_size / self.distance
        max_obs_pixel_size = beam_fwhm / 10.0
        regrid_scale = int(np.floor(max_obs_pixel_size / model_pixel_size))

        # ensure regrid_scale is even
        if regrid_scale % 2 != 0:
            regrid_scale = regrid_scale - 1
        self.obs_pixel_size = regrid_scale * model_pixel_size
        self.image_size = image_size
        obs_image_size = int(np.ceil((image_size / self.obs_pixel_size).to("").value))

        # ensure obs_image_size is odd
        if obs_image_size % 2 == 0:
            obs_image_size += 1

        # Pad edges of brightness_grid with zeros until its size is
        # an integer multiple of regrid_scale
        new_model_size = self.x_axis.size + (
            regrid_scale - (self.x_axis.size % regrid_scale)
        )

        # ensure new_model_size is odd
        if new_model_size % 2 == 0:
            new_model_size += 1
        pad = (new_model_size - self.x_axis.size) // 2
        brightness_grid = (
            np.pad(
                self.brightness_grid.to("mJy/arcsec2").value,
                ((pad, pad), (pad, pad), (0, 0)),
            )
            * u.mJy
            / u.arcsec ** 2
        )

        # Convert to flux per pixel, regrid
        flux_grid = brightness_grid * model_pixel_size ** 2.0
        flux_grid = (
            np.sum(
                flux_grid.to("mJy").value.reshape(
                    flux_grid.shape[0] // regrid_scale,
                    regrid_scale,
                    flux_grid.shape[1] // regrid_scale,
                    regrid_scale,
                    flux_grid.shape[2],
                ),
                axis=(1, 3),
            )
            * u.mJy
        )
        brightness_regrid = flux_grid / self.obs_pixel_size ** 2.0

        # generate noisy background
        self.obs_brightness_grid = (
            np.random.normal(
                loc=0.0,
                scale=noise.to("mJy/arcsec2").value,
                size=(obs_image_size, obs_image_size, self.velocity_axis.size),
            )
            * u.mJy
            / u.arcsec ** 2
        )

        # add nebula sky brightness to the center
        start = (obs_image_size - brightness_regrid.shape[0]) // 2
        self.obs_brightness_grid[
            start : start + brightness_regrid.shape[0],
            start : start + brightness_regrid.shape[1],
            :,
        ] += brightness_regrid

        # Convolve with beam. Loop over channel to conserve RAM.
        self.beam_fwhm = beam_fwhm
        beam_sigma = beam_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        conv_sigma = int(round((beam_sigma / self.obs_pixel_size).to("").value))
        self.obs_brightness_grid = (
            gaussian_filter(
                self.obs_brightness_grid.to("mJy/arcsec2"),
                sigma=(conv_sigma, conv_sigma, 0),
                mode="wrap",
            )
            * u.mJy
            / u.arcsec ** 2
        )

        # Convert units to mJy/beam
        beam_area = np.pi * beam_fwhm ** 2.0 / (4.0 * np.log(2.0))
        self.obs_brightness_grid *= beam_area

        # Save to FITS image
        # N.B. Need to transpose data to match FITS orientation
        hdu = fits.PrimaryHDU(self.obs_brightness_grid.to("mJy").value.T)
        hdu.header["OBJECT"] = "Model"
        hdu.header["BTYPE"] = "Intensity"
        hdu.header["BUNIT"] = "mJy/beam"
        hdu.header["BSCALE"] = 1.0
        hdu.header["BZERO"] = 0.0
        hdu.header["EQUINOX"] = 2.0e3
        hdu.header["RADESYS"] = "FK5"
        hdu.header["CTYPE1"] = "RA---SIN"
        hdu.header["CRVAL1"] = (image_size / 2.0).to("deg").value
        hdu.header["CDELT1"] = -self.obs_pixel_size.to("deg").value
        hdu.header["CRPIX1"] = 1
        hdu.header["CUNIT1"] = "DEG"
        hdu.header["CTYPE2"] = "DEC--SIN"
        hdu.header["CRVAL2"] = -(image_size / 2.0).to("deg").value
        hdu.header["CDELT2"] = self.obs_pixel_size.to("deg").value
        hdu.header["CRPIX2"] = 1
        hdu.header["CUNIT2"] = "DEG"
        hdu.header["CTYPE3"] = "VELO-LSR"
        hdu.header["CRVAL3"] = self.velocity_axis[0].to("km/s").value
        hdu.header["CDELT3"] = self.chan_size.to("km/s").value
        hdu.header["CRPIX3"] = 1
        hdu.header["CUNIT3"] = "km/s"
        hdu.header["RESTFRQ"] = self.rest_freq.to("Hz").value
        hdu.header["HISTORY"] = "Parameters of Model HII Region:"
        hdu.header["HISTORY"] = "electron density = " + str(self.electron_density)
        hdu.header["HISTORY"] = "electron temperature = " + str(self.electron_temp)
        hdu.header["HISTORY"] = "diameter = " + str(self.diameter)
        hdu.header["HISTORY"] = "distance = " + str(self.distance)
        hdu.header["HISTORY"] = "nonthermal FWHM line width = " + str(self.nt_fwhm_vel)
        hdu.header["HISTORY"] = "beam FWHM width = " + str(self.beam_fwhm)
        hdu.header["HISTORY"] = "noise = " + str(self.noise)
        hdu.header["HISTORY"] = "number of pixels in 3D grid = " + str(self.n_grid)
        hdu.header["HISTORY"] = "number of spectral channels = " + str(self.n_chan)
        hdu.header["HISTORY"] = "velocity range = " + str(self.vel_width)
        hdu.header["HISTORY"] = "image size = " + str(self.image_size)
        hdu.header["HISTORY"] = "rrl n = " + str(self.rrl_n)
        hdu.header["HISTORY"] = "rrl dn = " + str(self.rrl_dn)
        hdu.header["HISTORY"] = "kinematic model = " + str(self.kinematics)
        hdu.writeto(fitsfile, overwrite=True)

    def fit(
        self,
        center_fitsfile="center.fits",
        e_center_fitsfile="e_center.fits",
        fwhm_fitsfile="fwhm.fits",
        e_fwhm_fitsfile="e_fwhm.fits",
    ):
        """
        Fit a Gaussian to the spectra of each pixel, and save fit
        centers and FWHMs to FITS files.

        Inputs:
          center_fitsfile, e_center_fitsfile :: strings
            Filename of Gaussian center and error FITS images
          fwhm_fitsfile, e_fwhm_fitsfile :: strings
            Filename of Gaussian FWHM and error FITS image

        Returns: Nothing
        """
        # Diameter of nebula + beam FWHM (pixels)
        pixel_diameter = int(
            np.ceil(
                (
                    ((self.diameter / self.distance).to("") * u.rad + self.beam_fwhm)
                    / self.obs_pixel_size
                )
                .to("")
                .value
            )
        )
        center = np.ones(self.obs_brightness_grid.shape[0:2]) * np.nan
        e_center = np.ones(self.obs_brightness_grid.shape[0:2]) * np.nan
        fwhm = np.ones(self.obs_brightness_grid.shape[0:2]) * np.nan
        e_fwhm = np.ones(self.obs_brightness_grid.shape[0:2]) * np.nan
        xdata = self.velocity_axis.to("km/s").value

        # Loop over pixels, but only process those within diameter
        numx, numy = self.obs_brightness_grid.shape[0:2]
        for i in range(numx):
            for j in range(numy):
                if (i - numx / 2.0) ** 2.0 + (j - numy / 2.0) ** 2.0 > (
                    pixel_diameter / 2.0
                ) ** 2.0:
                    continue
                ydata = self.obs_brightness_grid[i, j].to("mJy").value
                amp_guess = ydata.max() - ydata.min()
                center_guess = xdata[ydata.argmax()]
                fwhm_guess = 25.0
                bg_guess = ydata.min()
                guess = [amp_guess, center_guess, fwhm_guess, bg_guess]
                try:
                    popt, pcov = curve_fit(gaussian, xdata, ydata, guess)
                except RuntimeError:
                    continue
                center[i, j] = popt[1]
                e_center[i, j] = np.sqrt(pcov[1, 1])
                fwhm[i, j] = popt[2]
                e_fwhm[i, j] = np.sqrt(pcov[2, 2])

        # Set up header
        header = fits.Header()
        header["OBJECT"] = "Model"
        header["BUNIT"] = "km/s"
        header["BSCALE"] = 1.0
        header["BZERO"] = 0.0
        header["EQUINOX"] = 2.0e3
        header["RADESYS"] = "FK5"
        header["CTYPE1"] = "RA---SIN"
        header["CRVAL1"] = (self.image_size / 2.0).to("deg").value
        header["CDELT1"] = -self.obs_pixel_size.to("deg").value
        header["CRPIX1"] = 1
        header["CUNIT1"] = "DEG"
        header["CTYPE2"] = "DEC--SIN"
        header["CRVAL2"] = -(self.image_size / 2.0).to("deg").value
        header["CDELT2"] = self.obs_pixel_size.to("deg").value
        header["CRPIX2"] = 1
        header["CUNIT2"] = "DEG"
        header["HISTORY"] = "Parameters of Model HII Region:"
        header["HISTORY"] = "electron density = " + str(self.electron_density)
        header["HISTORY"] = "electron temperature = " + str(self.electron_temp)
        header["HISTORY"] = "diameter = " + str(self.diameter)
        header["HISTORY"] = "distance = " + str(self.distance)
        header["HISTORY"] = "nonthermal FWHM line width = " + str(self.nt_fwhm_vel)
        header["HISTORY"] = "beam FWHM width = " + str(self.beam_fwhm)
        header["HISTORY"] = "noise = " + str(self.noise)
        header["HISTORY"] = "number of pixels in 3D grid = " + str(self.n_grid)
        header["HISTORY"] = "number of spectral channels = " + str(self.n_chan)
        header["HISTORY"] = "velocity range = " + str(self.vel_width)
        header["HISTORY"] = "image size = " + str(self.image_size)
        header["HISTORY"] = "rrl n = " + str(self.rrl_n)
        header["HISTORY"] = "rrl dn = " + str(self.rrl_dn)
        header["HISTORY"] = "kinematic model = " + str(self.kinematics)

        # Save center data to FITS image
        # N.B. Need to transpose data to match FITS orientation
        hdu = fits.PrimaryHDU(center.T, header=header)
        hdu.header["BTYPE"] = "CenterVLSR"
        hdu.writeto(center_fitsfile, overwrite=True)

        # Error
        hdu = fits.PrimaryHDU(e_center.T, header=header)
        hdu.header["BTYPE"] = "e_CenterVLSR"
        hdu.writeto(e_center_fitsfile, overwrite=True)

        # Save FWHM data to FITS image
        # N.B. Need to transpose data to match FITS orientation
        hdu = fits.PrimaryHDU(fwhm.T, header=header)
        hdu.header["BTYPE"] = "CenterFWHM"
        hdu.writeto(fwhm_fitsfile, overwrite=True)

        # Error
        hdu = fits.PrimaryHDU(e_fwhm.T)
        hdu.header["BTYPE"] = "e_CenterFWHM"
        hdu.writeto(e_fwhm_fitsfile, overwrite=True)
