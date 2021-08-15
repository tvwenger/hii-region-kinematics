"""
physics.py
Equations governing free-free and recombination line physics, and
other radiative transfer methods.

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
import astropy.units as u
import astropy.constants as c


def calc_lnbratio(freq, electron_temp):
    """
    Compute the natural logarithm of the ratio of impact parameters
    ln(bmax/bmin) at a given frequency. Equation from ERA
    (Condon & Ransom).

    Inputs:
      freq :: scalar (with units)
        Observed frequency
      electron_temp :: scalar (with units)
        Electron temperature

    Returns: lnbratio
      lnbratio :: scalar (unitless)
        Natural logarithm of ratio of impact parameters
    """
    lnbratio = (3.0 * c.k_B * electron_temp / c.m_e) ** 1.5
    lnbratio *= c.m_e / (2.0 * np.pi * c.e.gauss ** 2.0 * freq)
    lnbratio = np.log(lnbratio)
    return lnbratio


def calc_ffabscoeff(freq, electron_temp, electron_density):
    """
    Calculate the free-free absorption coefficient at a given
    frequency. Approximation assumes LTE and Rayleigh-Jeans limit.
    Equation from ERA (Condon & Ransom).

    Inputs:
      freq :: scalar (with units)
        Observed frequency
      electron_temp :: scalar (with units)
        Electron temperature
      electron_density :: scalar (with units)
        Electron density

    Returns: ffabscoeff
      ffabscoeff :: scalar (with units)
        Free-free absorption coefficient
    """
    lnbratio = calc_lnbratio(freq, electron_temp)
    ffabscoeff = 1.0 / (freq ** 2.0 * electron_temp ** (1.5))
    ffabscoeff *= c.e.gauss ** 6.0 * electron_density ** 2.0
    ffabscoeff /= c.c * np.sqrt(2.0 * np.pi * (c.m_e * c.k_B) ** 3.0)
    ffabscoeff *= np.pi ** 2.0 / 4.0 * lnbratio
    return ffabscoeff


def rydberg(rrl_n, delta_n=1):
    """
    Compute the rest frequency of a given hydrogen recombination
    transition.

    Inputs:
      rrl_n :: integer
        RRL principal quantum number
      delta_n :: integer
        Principal quantum number transition

    Returns: rest_freq
      rest_freq :: scalar (with units)
        Rest frequency of the transition
    """
    Ryd_M = c.Ryd / (1.0 + c.m_e / c.m_p)
    rest_freq = c.c * Ryd_M * (rrl_n ** -2.0 - (rrl_n + delta_n) ** -2.0)
    return rest_freq


def doppler(rest_freq, velocity):
    """
    Compute the Doppler-shifted line-center velocity in the
    non-relativistic limit.

    Inputs:
      rest_freq :: scalar (with units)
        Rest frequency
      velocity :: scalar (with units)
        Source velocity

    Returns: center_freq
      center_freq :: scalar (with units)
        Observed line center frequency
    """
    center_freq = rest_freq * (1.0 - velocity / c.c)
    return center_freq


def calc_thermalwidth(center_freq, electron_temp):
    """
    Compute the RRL thermal FWHM linewidth in frequency units.

    Inputs:
      center_freq :: scalar (with units)
        Observed line center frequency
      electron_temp :: scalar (with units)
        Electron temperature

    Returns:
      width :: scalar (with units)
        RRL thermal FWHM linewidth in frequency units
    """
    linewidth = np.sqrt(8.0 * np.log(2.0) * c.k_B / c.c ** 2.0)
    linewidth = linewidth * np.sqrt(electron_temp / c.m_p)
    linewidth = linewidth * center_freq
    return linewidth


def calc_line_profile(freq, center_freq, rrl_fwhm):
    """
    Calculate the normalized Gaussian RRL line profile at a
    given frequency.

    Inputs:
      freq :: scalar (with units)
        Observed frequency
      center_freq :: scalar (with units)
        Observed line center frequency
      rrl_fwhm :: scalar (with units)
        Frequency FWHM width of the RRL

    Returns: line_profile
      line_profile : scalar (with units)
        Line profile (inverse frequency units)
    """
    sigma = rrl_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    line_profile = np.exp(-0.5 * ((freq - center_freq) / sigma) ** 2.0)
    line_profile /= sigma * np.sqrt(2.0 * np.pi)
    return line_profile


def calc_rrlabscoeff(
    freq, rest_freq, center_freq, rrl_n, rrl_fwhm, electron_temp, electron_density
):
    """
    Calculate the RRL absorption coefficient at a given frequency,
    assuming LTE. Equation from ERA (Condon & Ransom).

    Inputs:
      freq :: scalar (with units)
        Observed frequency
      rest_freq :: scalar (with units)
        RRL rest frequency
      center_freq :: scalar (with units)
        Observed RRL center frequency
      rrl_n :: integer
        RRL principal quantum number
      rrl_fwhm :: scalar (with units)
        Frequency FWHM width of the RRL
      electron_temp :: scalar (with units)
        Electron temperature
      electron_density :: scalar (with units)
        Electron density
      velocity :: scalar (with units)
        Line-of-sight velocity of the source

    Returns: rrlabscoeff
      rrlabscoeff :: scalar (with units)
        RRL absorption coefficient
    """
    line_profile = calc_line_profile(freq, center_freq, rrl_fwhm)
    # ratio of statistical weights
    gratio = (rrl_n + 1.0) ** 2.0 / rrl_n ** 2.0
    # Saha equation, assuming exp(ionization_potential/kT) ~ 1.0
    Nn = rrl_n ** 2.0 * electron_density ** 2.0
    Nn *= (c.h ** 2.0 / (2.0 * np.pi * c.m_e * c.k_B * electron_temp)) ** (1.5)
    # spontaneous emission rate
    An = (
        64.0
        * np.pi ** 6.0
        * c.m_e
        * c.e.gauss ** 10.0
        / (3.0 * c.c ** 3.0 * c.h ** 6.0 * rrl_n ** 5.0)
    )
    # RRL absorption coefficient
    rrlabscoeff = (
        c.c ** 2.0 / (8.0 * np.pi * rest_freq ** 2.0) * gratio * Nn * An * line_profile
    )
    rrlabscoeff *= 1.0 - np.exp(-c.h * rest_freq / (c.k_B * electron_temp))
    return rrlabscoeff


def calc_optical_depth(abscoeffs, cell_size, axis=-1):
    """
    Compute the optical depth by integrating a grid of absorption
    coefficients along an axis.

    Inputs:
      abscoeffs :: N-dimnesional array of scalars (with units)
        Grid of absorption coefficients
      cell_size :: scalar (with units)
        Physical size of each grid cell
      axis :: integer
        Axis over which to sum

    Returns: tau
      tau :: (N-1)-dimensional array of scalars (unitless)
        Optical depth grid
    """
    tau = np.nansum(abscoeffs, axis=axis) * cell_size
    return tau


def calc_brightness(freq, optical_depth, electron_temp):
    """
    Compute the brightness (flux density per angular area) at a given
    frequency, in the Rayleigh-Jeans limit.

    Inputs:
      freq :: scalar (with units)
        Observed frequency
      optical_depth :: scalar (unitless)
        Optical depth
      electron_temp :: scalar (with units)
        Electron temperature

    Returns: brightness
      brightness :: scalar (with units)
        Brightness (flux density per angular area)
    """
    brightness_temp = electron_temp * (1.0 - np.exp(-optical_depth))
    brightness = 2.0 * c.k_B * brightness_temp * freq ** 2.0 / c.c ** 2.0 / u.sr
    return brightness
