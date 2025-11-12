"""
This script provides coordinate transformations from Geodetic -> ECEF, ECEF -> ENU
and Geodetic -> ENU (the composition of the two previous functions).

Also inverse transformation are available.

based on https://gist.github.com/govert/1b373696c9a27ff4c72a.
"""
import math

a = 6378137
b = 6356752.3142
f = (a - b) / a
e_sq = f * (2-f)

def latlon_to_ecef(lat, lon, h):
    # (lat, lon) in WSG-84 degrees
    # h in meters
    lamb = math.radians(lat)
    phi = math.radians(lon)
    s = math.sin(lamb)
    N = a / math.sqrt(1 - e_sq * s * s)

    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    x = (h + N) * cos_lambda * cos_phi
    y = (h + N) * cos_lambda * sin_phi
    z = (h + (1 - e_sq) * N) * sin_lambda

    return x, y, z

def ecef_to_enu(x, y, z, lat0, lon0, h0):
    lamb = math.radians(lat0)
    phi = math.radians(lon0)
    s = math.sin(lamb)
    N = a / math.sqrt(1 - e_sq * s * s)

    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - e_sq) * N) * sin_lambda

    xd = x - x0
    yd = y - y0
    zd = z - z0

    xEast = -sin_phi * xd + cos_phi * yd
    yNorth = -cos_phi * sin_lambda * xd - sin_lambda * sin_phi * yd + cos_lambda * zd
    zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd

    return xEast, yNorth, zUp

def latlon_to_enu(lat, lon, h, lat_ref, lon_ref, h_ref):
    x, y, z = latlon_to_ecef(lat, lon, h)
    
    return ecef_to_enu(x, y, z, lat_ref, lon_ref, h_ref)


def enu_to_ecef(xEast, yNorth, zUp, lat0, lon0, h0):

    # Convert to radians in notation consistent with the paper:
    lam = math.radians(lat0)
    phi = math.radians(lon0)
    s = math.sin(lam)
    N = a / math.sqrt(1 - e_sq * s * s)
    sin_lambda = math.sin(lam)
    cos_lambda = math.cos(lam)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - e_sq) * N) * sin_lambda

    xd = -sin_phi * xEast - cos_phi * sin_lambda * yNorth + cos_lambda * cos_phi * zUp
    yd = cos_phi * xEast - sin_lambda * sin_phi * yNorth + cos_lambda * sin_phi * zUp
    zd = cos_lambda * yNorth + sin_lambda * zUp

    x = xd + x0
    y = yd + y0
    z = zd + z0

    return x,y,z

def ecef_to_latlon(x, y, z):
    eps = e_sq / (1.0 - e_sq)
    p = math.sqrt(x * x + y * y)
    q = math.atan2((z * a), (p * b))
    sin_q = math.sin(q)
    cos_q = math.cos(q)
    sin_q_3 = sin_q * sin_q * sin_q
    cos_q_3 = cos_q * cos_q * cos_q
    phi = math.atan2((z + eps * b * sin_q_3), (p - e_sq * a * cos_q_3))
    lam = math.atan2(y, x)
    v = a / math.sqrt(1.0 - e_sq * math.sin(phi) * math.sin(phi))
    h = (p / math.cos(phi)) - v

    lat = math.degrees(phi)
    lon = math.degrees(lam)

    return lat, lon, h

def enu_to_latlon(x,y, lat0, lon0):
    x,y,z = enu_to_ecef(x, y, 0, lat0, lon0, 0)
    lat, lon, h = ecef_to_latlon(x, y, z)

    return lat,lon
