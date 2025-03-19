import astropy.units as u
from astropy.coordinates import AltAz, Angle, EarthLocation, SkyCoord
from astropy.time import Time

def pseudo_parallactic_angle(
    ra: float,
    dec: float,
    mjd: float,
    lon: float = -70.7494,
    lat: float = -30.2444,
    height: float = 2650.0,
    pressure: float = 750.0,
    temperature: float = 11.5,
    relative_humidity: float = 0.4,
    obswl: float = 1.0,
):
    """Compute the pseudo parallactic angle.

    The (traditional) parallactic angle is the angle zenith - coord - NCP
    where NCP is the true-of-date north celestial pole.  This function instead
    computes zenith - coord - NCP_ICRF where NCP_ICRF is the north celestial
    pole in the International Celestial Reference Frame.

    Parameters
    ----------
    ra, dec : float
        ICRF coordinates in degrees.
    mjd : float
        Modified Julian Date.
    latitude, longitude : float
        Geodetic coordinates of observer in degrees.
    height : float
        Height of observer above reference ellipsoid in meters.
    pressure : float
        Atmospheric pressure in millibars.
    temperature : float
        Atmospheric temperature in degrees Celsius.
    relative_humidity : float
    obswl : float
        Observation wavelength in microns.

    Returns
    -------
    ppa : float
        The pseudo parallactic angle in degrees.
    """
    obstime = Time(mjd, format="mjd", scale="tai")
    location = EarthLocation.from_geodetic(
        lon=lon * u.deg,
        lat=lat * u.deg,
        height=height * u.m,
        ellipsoid="WGS84",  # For concreteness
    )

    coord_kwargs = dict(
        obstime=obstime,
        location=location,
        pressure=pressure * u.mbar,
        temperature=temperature * u.deg_C,
        relative_humidity=relative_humidity,
        obswl=obswl * u.micron,
    )

    coord = SkyCoord(ra * u.deg, dec * u.deg, **coord_kwargs)

    towards_zenith = SkyCoord(
        alt=coord.altaz.alt + 10 * u.arcsec,
        az=coord.altaz.az,
        frame=AltAz,
        **coord_kwargs
    )

    towards_north = SkyCoord(
        ra=coord.icrs.ra, dec=coord.icrs.dec + 10 * u.arcsec, **coord_kwargs
    )

    ppa = coord.position_angle(towards_zenith) - coord.position_angle(towards_north)
    return ppa.wrap_at(180 * u.deg).deg


def rtp_to_rsp(rotTelPos: float, ra: float, dec: float, mjd: float, **kwargs: dict):
    """Convert RotTelPos -> RotSkyPos.

    Parameters
    ----------
    rotTelPos : float
        Camera rotation angle in degrees.
    ra, dec : float
        ICRF coordinates in degrees.
    mjd : float
        Modified Julian Date.
    **kwargs : dict
        Other keyword arguments to pass to pseudo_parallactic_angle.  Defaults
        are generally appropriate for Rubin Observatory.

    Returns
    -------
    rsp : float
        RotSkyPos in degrees.
    """
    q = pseudo_parallactic_angle(ra, dec, mjd, **kwargs)
    return Angle((270 - rotTelPos + q)*u.deg).wrap_at(180 * u.deg).deg


def rsp_to_rtp(rotSkyPos: float, ra: float, dec: float, mjd: float, **kwargs: dict):
    """Convert RotTelPos -> RotSkyPos.

    Parameters
    ----------
    rotSkyPos : float
        Sky rotation angle in degrees.
    ra, dec : float
        ICRF coordinates in degrees.
    mjd : float
        Modified Julian Date.
    **kwargs : dict
        Other keyword arguments to pass to pseudo_parallactic_angle.  Defaults
        are generally appropriate for Rubin Observatory.

    Returns
    -------
    rsp : float
        RotSkyPos in degrees.
    """
    q = pseudo_parallactic_angle(ra, dec, mjd, **kwargs)
    return Angle((270 - rotSkyPos + q)*u.deg).wrap_at(180 * u.deg)