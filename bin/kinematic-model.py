#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kinematic-model
Primary executable

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

import os
import argparse
import textwrap
import astropy.units as u
from kinematics import kinematic_model

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="HII Region Kinematic Models",
        prog="kinematic_model.py",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    PARSER.add_argument(
        "modelname", type=str, help="Model name, added to saved FITS images"
    )
    PARSER.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Output directory for FITS images (Default: current directory)",
    )
    PARSER.add_argument(
        "--density",
        type=float,
        default=250.0,
        help="Electron density (cm-3; default 250)",
    )
    PARSER.add_argument(
        "--temperature",
        type=float,
        default=8000.0,
        help="Electron temperature (K; default 8000)",
    )
    PARSER.add_argument(
        "--diameter", type=float, default=2.0, help="Diameter (pc; default 2.0)"
    )
    PARSER.add_argument(
        "--distance", type=float, default=5.0, help="Distance (kpc; default 5.0)"
    )
    PARSER.add_argument(
        "--nonthermal",
        type=float,
        default=15.0,
        help="Non-thermal FWHM line width (km/s; default 15.0)",
    )
    PARSER.add_argument(
        "--beam",
        type=float,
        default=90.0,
        help="Beam FWHM width (arcsec; default 90.0)",
    )
    PARSER.add_argument(
        "--noise",
        type=float,
        default=0.001,
        help="Pre-convolution image noise (mJy/arcsec2; default 0.001)",
    )
    PARSER.add_argument(
        "--grid",
        type=int,
        default=128,
        help="Number of points along model grid side (default 128)",
    )
    PARSER.add_argument(
        "--nchan",
        type=int,
        default=256,
        help="Number of velocity channels (default 256)",
    )
    PARSER.add_argument(
        "--velwidth",
        type=float,
        default=200.0,
        help="Full velocity range, centered at 0 (km/s; default 200.0)",
    )
    PARSER.add_argument(
        "--imagesize",
        type=float,
        default=600.0,
        help="Image width (arcsec; default 600.0)",
    )
    PARSER.add_argument(
        "--rrl", type=int, default=85, help="RRL principal quantum number (default 85)"
    )
    PARSER.add_argument(
        "--deltan",
        type=int,
        default=1,
        help="RRL principal quantum number transition (default 1)",
    )
    PARSER.add_argument(
        "--kinematic",
        action="append",
        nargs="+",
        default=[],
        help=textwrap.dedent(
            """
                        Add a kinematic component. Kinematics are applied
                        in the order passed, like:
                        --kinematic type1 <args1> --kinematic type2 <args2>

                        Options:
                        --kinematic solidbody <eq_speed> <sky_pa> <los_pa>
                        where <eq_speed> is equatorial rotation speed (km/s)
                        <sky_pa> and <los_pa> define angular momentum vector (deg)

                        --kinematic differential <eq_speed> <r_power> <sky_pa> <los_pa>
                        where <eq_speed> is equatorial rotation speed at surface (km/s)
                        <r_power> is exponent of radial change in rotation speed
                        <sky_pa> and <los_pa> define angular momentum vector (deg)

                        --kinematic outflow <speed> <angle> <sky_pa> <los_pa>
                        where <speed> is radial outflow speed (km/s)
                        <angle> is opening angle (deg)
                        <sky_pa> and <los_pa> define positive velocity outflow axis"""
        ),
    )
    PARSER.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing FITS images"
    )
    ARGS = vars(PARSER.parse_args())

    # Setup kinematic dictionaries
    kinematics = []
    for kinematic in ARGS["kinematic"]:
        if kinematic[0] == "solidbody":
            kinematics.append(
                {
                    "type": kinematic[0],
                    "eq_speed": float(kinematic[1]) * u.km / u.s,
                    "sky_pa": float(kinematic[2]) * u.deg,
                    "los_pa": float(kinematic[3]) * u.deg,
                }
            )
        elif kinematic[0] == "differential":
            kinematics.append(
                {
                    "type": kinematic[0],
                    "eq_speed": float(kinematic[1]) * u.km / u.s,
                    "r_power": float(kinematic[2]),
                    "sky_pa": float(kinematic[3]) * u.deg,
                    "los_pa": float(kinematic[4]) * u.deg,
                }
            )
        elif kinematic[0] == "outflow":
            kinematics.append(
                {
                    "type": kinematic[0],
                    "speed": float(kinematic[1]) * u.km / u.s,
                    "angle": float(kinematic[2]) * u.deg,
                    "sky_pa": float(kinematic[3]) * u.deg,
                    "los_pa": float(kinematic[4]) * u.deg,
                }
            )
        elif kinematic[0] == "expansion":
            kinematics.append(
                {
                    "type": kinematic[0],
                    "alpha": float(kinematic[1]) * u.km / u.s,
                    "beta": float(kinematic[2]),
                    "sky_pa": 0.0 * u.deg,
                    "los_pa": 0.0 * u.deg,
                }
            )
        else:
            raise ValueError("Invalid kinematic type: {0}".format(kinematic[0]))

    # Setup output
    if not os.path.isdir(ARGS["outdir"]):
        os.mkdir(ARGS["outdir"])
    velocity_fits = os.path.join(
        ARGS["outdir"], "{0}_velocity.fits".format(ARGS["modelname"])
    )
    if os.path.exists(velocity_fits) and not ARGS["overwrite"]:
        raise IOError("Will not overwrite {0}".format(velocity_fits))
    sky_brightness_fits = os.path.join(
        ARGS["outdir"], "{0}_true.fits".format(ARGS["modelname"])
    )
    if os.path.exists(sky_brightness_fits) and not ARGS["overwrite"]:
        raise IOError("Will not overwrite {0}".format(sky_brightness_fits))
    obs_brightness_fits = os.path.join(
        ARGS["outdir"], "{0}_obs.fits".format(ARGS["modelname"])
    )
    if os.path.exists(obs_brightness_fits) and not ARGS["overwrite"]:
        raise IOError("Will not overwrite {0}".format(obs_brightness_fits))
    center_fits = os.path.join(
        ARGS["outdir"], "{0}_fit_center.fits".format(ARGS["modelname"])
    )
    if os.path.exists(center_fits) and not ARGS["overwrite"]:
        raise IOError("Will not overwrite {0}".format(center_fits))
    e_center_fits = os.path.join(
        ARGS["outdir"], "{0}_fit_e_center.fits".format(ARGS["modelname"])
    )
    if os.path.exists(e_center_fits) and not ARGS["overwrite"]:
        raise IOError("Will not overwrite {0}".format(center_fits))
    fwhm_fits = os.path.join(
        ARGS["outdir"], "{0}_fit_fwhm.fits".format(ARGS["modelname"])
    )
    if os.path.exists(fwhm_fits) and not ARGS["overwrite"]:
        raise IOError("Will not overwrite {0}".format(fwhm_fits))
    e_fwhm_fits = os.path.join(
        ARGS["outdir"], "{0}_fit_e_fwhm.fits".format(ARGS["modelname"])
    )
    if os.path.exists(e_fwhm_fits) and not ARGS["overwrite"]:
        raise IOError("Will not overwrite {0}".format(fwhm_fits))

    # Run
    kinematic_model.main(
        ARGS["density"],
        ARGS["temperature"],
        ARGS["diameter"],
        kinematics=kinematics,
        grid_size=ARGS["grid"],
        num_channels=ARGS["nchan"],
        vel_width=ARGS["velwidth"],
        rrl_n=ARGS["rrl"],
        delta_n=ARGS["deltan"],
        nonthermal_fwhm=ARGS["nonthermal"],
        distance=ARGS["distance"],
        beam_fwhm=ARGS["beam"],
        image_size=ARGS["imagesize"],
        noise=ARGS["noise"],
        velocity_fits=velocity_fits,
        sky_brightness_fits=sky_brightness_fits,
        obs_brightness_fits=obs_brightness_fits,
        center_fits=center_fits,
        e_center_fits=e_center_fits,
        fwhm_fits=fwhm_fits,
        e_fwhm_fits=e_fwhm_fits,
    )
