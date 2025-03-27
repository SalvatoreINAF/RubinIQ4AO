import os
import numpy as np
from math import tan, radians, sqrt
from functools import reduce
from astropy.table import Table
import matplotlib.pyplot as plt

def refractive_index_formula(wavelength, pressure_hPa=799.932, temperature=7, water_vapor_pressure_hPa=10.66576, method='filippenko'):
    """
    Calculate the refractive index of air using the Filippenko formula.
    - wavelength: Wavelength of light in micrometers (µm).
    - pressure: Local atmospheric pressure in hPa (default: 799.932 hPa (= 600 mmHg) pressure @2km altitude).
    - temperature: Local temperature in degrees Celsius (default: 7°C).
    - water_vapor_pressure: Water vapor pressure in hPa (default: 10.66576 hPa (= 8 hPa)).
    - Refractive index of air (dimensionless).
    """

    if method != 'filippenko':
        raise ValueError(f"Invalid method: {method}")

    pressure = hpa2mmhg(pressure_hPa)  # Convert pressure to mmHg
    water_vapor_pressure = hpa2mmhg(water_vapor_pressure_hPa)  # Convert water vapor pressure to mmHg

    c1 = 64.328
    c2 = 29498.1 / (146.0 - wavelength**(-2))
    c3 = 255.4 / (41.0 - wavelength**(-2))

    n_sea_level = c1 + c2 + c3

    water_vapor_term = water_vapor_pressure * (0.0624 - 0.000680*wavelength**(-2)) / (1 + 0.003661*temperature)

    n = n_sea_level * ( pressure * (1 + (1.049-0.0157*temperature) * pressure * 1e-6) ) / ( 720.883 * (1 + 0.003661*temperature) ) - water_vapor_term

    return (1 + n/1e6)

def mmhg2hpa(P_mmhg):
    return P_mmhg * 1.33322

def hpa2mmhg(P_hpa):
    return P_hpa / 1.33322

def compute_atm_dispersion(zenith_angle, filter, pressure=800, temperature=15):
    """
    Compute the atmospheric dispersion correction factor for a telescope's second moment.

    Parameters:
    - zenith_angle: Scalar or numpy array, zenith angle in degrees.
    - filter: String, optical filter ('g', 'i', 'r', 'u', 'y', 'z').
    - pressure: Atmospheric pressure in hPa (default: 800 hPa).
    - temperature: Temperature in Celsius (default: 15 C).

    Returns:
    - Correction factor (correction) in arcseconds^2.
    """
    
    # Ensure zenith_angle is a numpy array
    zenith_angle = np.asarray(zenith_angle)

    if np.any(zenith_angle < 5):
        raise ValueError("There are angles below 5 degrees. Are you sure they are in degrees?")

    # Convert zenith angle to radians
    z = np.radians(zenith_angle)

    # Rubin Filter-specific parameters
    filters_dictionary = {
        'g': {'bounds': (400.9115872633698, 551.0628186080594), 'second_moment': 1928.0241695984676},
        'i': {'bounds': (691.5407557548015, 821.0438077701019), 'second_moment': 1498.8626356736725},
        'r': {'bounds': (549.2226726857816, 691.1169465126783), 'second_moment': 1719.7907111676918},
        'u': {'bounds': (345.39460683030137, 395.173672637585), 'second_moment': 654.8287680639575},
        'y': {'bounds': (925.9602208876246, 1001.509470055575), 'second_moment': 1275.9609802400944},
        'z': {'bounds': (817.4316768076759, 922.5347115628629), 'second_moment': 1029.913295671198},
    }

    if filter not in filters_dictionary:
        raise ValueError(f"Invalid filter: {filter}")
    
    lambda_min, lambda_max = filters_dictionary[filter]['bounds']
    mu_f = filters_dictionary[filter]['second_moment']

    # Compute refractive indices using Filippenko formula
    n_lambda_min = refractive_index_formula(lambda_min * 1e-3, pressure, temperature)
    n_lambda_max = refractive_index_formula(lambda_max * 1e-3, pressure, temperature)

    # Compute delta and gamma
    delta = (n_lambda_min - n_lambda_max) * np.tan(z) * 206265.  # [arcsec]
    gamma = sqrt(mu_f) / (lambda_max - lambda_min)

    # Compute correction factor
    correction = gamma**2 * delta**2  # [arcsec]^2

    return correction

def addADC_to_Table(table, elevation_angle, filter, pressure=800, temperature=15):
    """
    Add the atmospheric dispersion correction factor to the icSrc table.
    The correction is in place.

    Parameters:
    - table: icSrc table containing the moments, in particular "aa_Iyy".
    - elevation_angle: Scalar or numpy array, elevation angle in degrees.
    - filter: String, optical filter ('g', 'i', 'r', 'u', 'y', 'z').
    - pressure: Atmospheric pressure in hPa (default: 800 hPa).
    - temperature: Temperature in Celsius (default: 15 C).

    Returns:
    - Corrected table.
    """

    # Check if the column 'aa_Iyy' exists
    if 'aa_Iyy' not in table.colnames:
        raise ValueError("Column 'aa_Iyy' not found in the table.")
    
    # Check if elevation_angle is a scalar (not list, array, etc.)
    if not isinstance(elevation_angle, (int, float)):
        raise TypeError("Elevation_angle must be a scalar (int or float), not a list or array.")

    # Compute the correction factor
    zenith_angle = 90 - elevation_angle
    correction_factor = compute_atm_dispersion(zenith_angle, filter, pressure, temperature)

    # Create columns for the uncorrected moments
    table['aa_unc_Ixx'] = table['aa_Ixx']
    table['aa_unc_Ixy'] = table['aa_Ixy']
    table['aa_unc_Iyy'] = table['aa_Iyy']
    table['aa_unc_e1'] = table['aa_e1']
    table['aa_unc_e2'] = table['aa_e2']
    table['aa_unc_x'] = table['aa_x']
    table['aa_unc_y'] = table['aa_y']

    # calculate the ellipticities with uncorrected moments
    table["aa_unc_T"] = table["aa_Ixx"] + table["aa_unc_Iyy"]
    table["aa_unc_e1"] = (table["aa_unc_Ixx"] - table["aa_unc_Iyy"]) / table["aa_unc_T"]
    table["aa_unc_e2"] = 2 * table["aa_unc_Ixy"] / table["aa_unc_T"]
    table["aa_unc_e"] = np.hypot(table["aa_unc_e1"], table["aa_unc_e2"])
    
    # Add the correction factor to the 'aa_Iyy' column in place
    table['aa_Iyy'] = table['aa_unc_Iyy'] - correction_factor

    # Recalculate the ellipticities with the corrected yy component
    table["aa_T"] = table["aa_Ixx"] + table["aa_Iyy"]
    table["aa_e1"] = (table["aa_Ixx"] - table["aa_Iyy"]) / table["aa_T"]
    table["aa_e2"] = 2 * table["aa_Ixy"] / table["aa_T"]
    table["aa_e"] = np.hypot(table["aa_e1"], table["aa_e2"])

    return table
