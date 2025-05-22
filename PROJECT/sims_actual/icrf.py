import pandas as pd
import numpy as np
from astropy.coordinates import CartesianRepresentation, SkyCoord
from astropy.time import Time
from astropy import units as u

def to_icrf(x, y, z, epoch):
    """Convert heliocentric ecliptic coordinates to ICRF"""
    # Create Cartesian representation with units
    cartrep = CartesianRepresentation(
        x=np.array(x)*u.AU,
        y=np.array(y)*u.AU,
        z=np.array(z)*u.AU
    )
    
    # Create SkyCoord in heliocentric MEAN ecliptic frame (J2000)
    ecliptic_coords = SkyCoord(
        cartrep,
        frame='heliocentricmeanecliptic',
        obstime=epoch,
        equinox='J2000'
    )
    
    # Transform to ICRS
    icrf_coords = ecliptic_coords.transform_to('icrs')
    return (
        icrf_coords.cartesian.x.value,
        icrf_coords.cartesian.y.value,
        icrf_coords.cartesian.z.value
    )

# Load simulation data
df = pd.read_csv("sims_actual/2body.csv")

# Convert coordinates
try:
    x_icrf, y_icrf, z_icrf = to_icrf(df["x"], df["y"], df["z"], Time('J2000'))
    
    # Create transformed DataFrame
    df_icrf = df.copy()
    df_icrf["x"] = x_icrf
    df_icrf["y"] = y_icrf
    df_icrf["z"] = z_icrf
    
    # Save results
    df_icrf.to_csv("sims_actual/2bodyicrf.csv", index=False)
    print("Successfully saved ICRF-transformed coordinates to 2bodyicrf.csv")
    print("\nFirst transformed point:")
    print(f"X: {x_icrf[0]:.6f} AU")
    print(f"Y: {y_icrf[0]:.6f} AU") 
    print(f"Z: {z_icrf[0]:.6f} AU")

except Exception as e:
    print(f"Error during coordinate transformation: {str(e)}")