import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.coordinates import solar_system_ephemeris, get_body
from sunpy.coordinates import frames as sframes

def get_moon_position(t: str, frame = 'HEE'):
    """
        Get the Moon's position in HEE coordinates at a given list of times.
    """

    # Choose a time (UTC). Can be an array of times as well.
    time = Time(t, scale="utc")


    # 1) Moon position in GCRS (geocentric), as seen from Earth’s center
    moon_gcrs = get_body('moon', time)  # SkyCoord in GCRS

    if frame == 'GSE':
        # Transform to GSE (Earth-centered; X->Sun, Z->ecliptic pole) and get Cartesian xyz
        moon = moon_gcrs.transform_to(
            sframes.GeocentricSolarEcliptic(obstime=t, representation_type="cartesian")
        )

    elif frame == 'HEE':
            # 2) Transform to GSE (Earth-centered; X->Sun, Z->ecliptic pole) and get Cartesian xyz
        moon = moon_gcrs.transform_to(
            sframes.HeliocentricEarthEcliptic(obstime=t, representation_type="cartesian")
        )

    else:
        raise ValueError("Invalid frame. Choose 'GSE' or 'HEE'.")

    # Values in kilometers:
    x_km = moon.cartesian.x.to(u.km).value
    y_km = moon.cartesian.y.to(u.km).value
    z_km = moon.cartesian.z.to(u.km).value

    # Values in RE:
    x_re = x_km / 6378.137
    y_re = y_km / 6378.137
    z_re = z_km / 6378.137

    return (x_re, y_re, z_re)
    

def GSE_to_HEE(pos, t: str):
    """
        Transform coordinates from Geocentric Solar Ecliptic (GSE) to Heliocentric Earth Ecliptic (HEE).
    """

    # unpacjk position
    x_re, y_re, z_re = pos

    # Convert RE to km
    x_km = x_re * 6378.137
    y_km = y_re * 6378.137
    z_km = z_re * 6378.137

    # Create SkyCoord in GSE
    coordinates = SkyCoord(x = x_km * u.km, y = y_km * u.km, z = z_km * u.km, frame=sframes.GeocentricSolarEcliptic, representation_type='cartesian', obstime=t)
    # Transform to HEE
    coordinates_HEE = coordinates.transform_to(sframes.HeliocentricEarthEcliptic(obstime=t, representation_type='cartesian'))

    # Values in kilometers:
    x_km = coordinates_HEE.cartesian.x.to(u.km).value
    y_km = coordinates_HEE.cartesian.y.to(u.km).value
    z_km = coordinates_HEE.cartesian.z.to(u.km).value

    # Values in RE:
    x_re = x_km / 6378.137
    y_re = y_km / 6378.137
    z_re = z_km / 6378.137

    return (x_re, y_re, z_re)


# Rotate moon position x degrees around HEE z-axis then y-axis
def rotate_moon(pos, angle_deg_y, angle_deg_z):
    """
        Rotate the Moon's position around the HEE z-axis first, then around the y-axis by given angles in degrees.
    """
    # Unpack position
    x_re, y_re, z_re = pos

    # Convert angles to radians
    angle_rad_z = np.radians(angle_deg_z)
    angle_rad_y = np.radians(angle_deg_y)

    # Rotation matrix around z-axis
    R_z = np.array([
        [np.cos(angle_rad_y), np.sin(angle_rad_y), 0],
        [-np.sin(angle_rad_y),  np.cos(angle_rad_y), 0],
        [0,                   0,                   1]
    ])

    # Rotation matrix around y-axis
    R_y = np.array([
        [np.cos(angle_rad_z),  0, np.sin(angle_rad_z)],
        [0,                   1, 0],
        [-np.sin(angle_rad_z), 0, np.cos(angle_rad_z)]
    ])

    # Original position vector
    pos_vector = np.array([x_re, y_re, z_re])

    # Apply rotations: first z-axis, then y-axis
    rotated_vector = R_y @ (R_z @ pos_vector)

    return tuple(rotated_vector)

def calculate_rel_sat_pos(moon_pos, sat_pos):
    """
        Calculate the satellite's position relative to the Moon.
    """

    # Unpack positions
    moon_x, moon_y, moon_z = moon_pos
    sat_x, sat_y, sat_z = sat_pos

    # Calculate relative position
    rel_x = sat_x - moon_x
    rel_y = sat_y - moon_y
    rel_z = sat_z - moon_z

    return (rel_x, rel_y, rel_z)

def calculate_unit_vector(pos):
    """
        Calculate the unit vector of a given position vector.
    """

    # Unpack position
    x, y, z = pos

    # Calculate magnitude
    magnitude = np.sqrt(x**2 + y**2 + z**2)

    if magnitude == 0:
        raise ValueError("Cannot compute unit vector for zero vector.")

    # Calculate unit vector
    unit_vector = (x / magnitude, y / magnitude, z / magnitude)

    return unit_vector

def check_lunar_wake(moon_vect, sat_vect):
    """
        Check if the satellite is in the lunar wake region.
        The lunar wake is defined as the region behind the Moon (relative to the Sun) within 5 RE.
    """

    # Convert to numpy arrays
    moon_unit_vect = calculate_unit_vector(moon_vect)
    sat_unit_vect = calculate_unit_vector(sat_vect)

    # Calculate the angle between the two vectors using the dot product
    dot_product = np.dot(moon_unit_vect, sat_unit_vect)
    angle_rad = np.arccos(dot_product)

    # Calculate angle of lunar radius as seem from the sun
    lunar_radius_re = 1737.4 / 6378.137 * 1.1 # Lunar radius in RE (with a 20% margin)
    angle_lunar_rad = np.arctan(lunar_radius_re / (149597870.700 / 6378.137))  # 1 AU in RE

    # Calculate distance from moon to satellite
    distance_re = np.sqrt((sat_vect[0] - moon_vect[0])**2 + (sat_vect[1] - moon_vect[1])**2 + (sat_vect[2] - moon_vect[2])**2)

    # Check whether the satellise is in front of or behind the moon (relative to the sun)
    mag_moon = np.sqrt(moon_vect[0]**2 + moon_vect[1]**2 + moon_vect[2]**2)
    mag_sat = np.sqrt(sat_vect[0]**2 + sat_vect[1]**2 + sat_vect[2]**2)

    if mag_sat < mag_moon:
        return False  # In front of the moon, cannot be in lunar wake

    # Increase lunar wake angle depending on distance
    if distance_re < 100:
        angle_lunar_rad *= (distance_re) / 10 * 0.1 + 1  # Linearly increase angle up to double at 0 RE


    # Check if the angle is within the lunar wake cone
    if angle_rad < angle_lunar_rad:
        return True
    else:
        return False
    
def lunar_wake_check(sat_pos, t: str, angle_deg_y, angle_deg_z):
    """
        Check if the satellite is in the lunar wake (from the solar wind) region at a given time.
    """

    # Get Moon position in HEE
    moon_pos_HEE = get_moon_position(t)

    # Transform satellite position from GSE to HEE
    sat_pos_HEE = GSE_to_HEE(sat_pos, t)

    # Calculate relative position of satellite to Moon
    rel_sat_pos = calculate_rel_sat_pos(moon_pos_HEE, sat_pos_HEE)

    # Rotate Moon position to account for Earth's orbit around the Sun, to align with solar wind direction
    moon_pos_HEE_rotated = rotate_moon(moon_pos_HEE, angle_deg_y, angle_deg_z)  # Approximate tilt of Earth's axis

    # Calculate satellite position after moon's rotation
    sat_pos_HEE_rotated = np.array(moon_pos_HEE_rotated) + np.array(rel_sat_pos)

    # Check if satellite is in lunar wake
    in_lunar_wake = check_lunar_wake(moon_pos_HEE_rotated, sat_pos_HEE_rotated)

    return in_lunar_wake

def lunar_wake_check_series(sat_pos_series, time_series, aberration_angle_y_series, aberration_angle_z_series):
    """
        Check if the satellite is in the lunar wake (from the solar wind) region for a series of positions and times.
    """

    if len(sat_pos_series) != len(time_series):
        raise ValueError("sat_pos_series and time_series must have the same length.")

    results = []
    for sat_pos, t, aberration_angle_y, aberration_angle_z in zip(sat_pos_series, time_series, aberration_angle_y_series, aberration_angle_z_series):
        result = lunar_wake_check(sat_pos, t, aberration_angle_y, aberration_angle_z)
        results.append(result)

    return results