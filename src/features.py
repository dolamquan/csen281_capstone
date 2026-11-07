"""Shared data loading and feature engineering for the wildfire capstone.

Builds one row per fire with the target (FIRE_SIZE >= 100 acres), discovery-time
controls and fire-station features, and caches it as parquet so notebooks do not
redo the 2.6M-row station join.
"""
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree

EARTH_KM = 6371.0
JD_EPOCH = 2440587.5          # julian day number of 1970-01-01
LARGE_ACRES = 100             # size class D and above
STACK_MIN = 200               # an exact coordinate shared by this many fires = office / placeholder
STATION_RADIUS_KM = 25
CAUSE_MISSING = 'Missing data/not specified/undetermined'

REGION6 = {**dict.fromkeys(['CA', 'OR', 'WA'], 'Pacific'),
           **dict.fromkeys(['ID', 'NV', 'UT', 'MT', 'WY', 'CO'], 'Mountain West'),
           **dict.fromkeys(['AZ', 'NM'], 'Southwest'),
           **dict.fromkeys(['TX', 'OK', 'KS', 'NE', 'SD', 'ND'], 'Plains'),
           **dict.fromkeys(['AL', 'AR', 'FL', 'GA', 'KY', 'LA', 'MS', 'NC', 'SC', 'TN', 'VA', 'WV'], 'Southeast'),
           'AK': 'Alaska'}
NOT_MAINLAND = {'HI', 'PR'}

FIRE_COLUMNS = ['FIRE_YEAR', 'DISCOVERY_DATE', 'DISCOVERY_DOY', 'DISCOVERY_TIME', 'FIRE_SIZE', 'LATITUDE',
                'LONGITUDE', 'STATE', 'OWNER_DESCR', 'NWCG_CAUSE_CLASSIFICATION']


def repo_root(start=None):
    """Walk up from `start` (default: cwd) to the folder that holds the data directories."""
    start = Path(start or Path.cwd()).resolve()
    for d in [start, *start.parents]:
        if (d / 'wildfire_7th_edition_data').is_dir():
            return d
    raise FileNotFoundError('wildfire_7th_edition_data/ not found above ' + str(start))


def data_paths(root=None):
    root = repo_root(root)
    db = root / 'wildfire_7th_edition_data' / 'Data' / 'FPA_FOD_20260615.sqlite'
    stations = root / 'fire_station_data' / 'Data' / 'fire-and-emergency-medical-service-ems-stations-geojson.geojson'
    for p in (db, stations):
        if not p.is_file():
            raise FileNotFoundError(f'{p} is missing, see the README in that data folder for the download link')
    return db, stations


def load_fires(db_path):
    with closing(sqlite3.connect(db_path)) as conn:
        return pd.read_sql_query('SELECT ' + ', '.join(FIRE_COLUMNS) + ' FROM Fires', conn)


def load_stations(stations_path):
    with open(stations_path, encoding='utf-8') as fh:
        features = json.load(fh)['features']
    return pd.DataFrame({'lat': [f['geometry']['coordinates'][1] for f in features],
                         'lon': [f['geometry']['coordinates'][0] for f in features]})


def _unit_xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def station_features(lat, lon, stations, k=3, radius_km=STATION_RADIUS_KM):
    """Great-circle distance (km) to the k nearest stations and the station count within `radius_km`."""
    tree = BallTree(np.radians(stations[['lat', 'lon']].to_numpy()), metric='haversine')
    dist, _ = tree.query(np.radians(np.column_stack([lat, lon])), k=k)
    out = {f'D{i + 1}_KM': dist[:, i] * EARTH_KM for i in range(k)}
    chord = 2 * np.sin(radius_km / (2 * EARTH_KM))
    kd = cKDTree(_unit_xyz(stations['lat'].to_numpy(), stations['lon'].to_numpy()))
    out[f'STATIONS_{radius_km}KM'] = kd.query_ball_point(_unit_xyz(lat, lon), r=chord, return_length=True, workers=-1)
    return pd.DataFrame(out)


def build_table(fires, stations):
    """One row per fire: target, outcome-free controls, station features and grouping columns."""
    state = fires['STATE'].str.upper()  # Oklahoma appears as both 'OK' and 'ok'
    date = pd.to_datetime(fires['DISCOVERY_DATE'] - JD_EPOCH, unit='D')
    hhmm = pd.to_numeric(fires['DISCOVERY_TIME'], errors='coerce')
    doy_angle = 2 * np.pi * fires['DISCOVERY_DOY'] / 365.25
    stacked = fires.groupby(['LATITUDE', 'LONGITUDE'])['FIRE_YEAR'].transform('size') >= STACK_MIN
    st = station_features(fires['LATITUDE'].to_numpy(), fires['LONGITUDE'].to_numpy(), stations)

    table = pd.DataFrame({
        'FIRE_YEAR': fires['FIRE_YEAR'].astype('int16'),
        'LARGE': (fires['FIRE_SIZE'] >= LARGE_ACRES).astype('int8'),
        'FIRE_SIZE': fires['FIRE_SIZE'],
        'LATITUDE': fires['LATITUDE'],
        'LONGITUDE': fires['LONGITUDE'],
        'STATE': state,
        'REGION': state.map(REGION6).fillna('North/East').where(~state.isin(NOT_MAINLAND), 'HI/PR'),
        'OWNER': fires['OWNER_DESCR'],
        'CAUSE': fires['NWCG_CAUSE_CLASSIFICATION'].replace({CAUSE_MISSING: 'Missing'}),
        'DOY_SIN': np.sin(doy_angle),
        'DOY_COS': np.cos(doy_angle),
        'WEEKDAY': date.dt.dayofweek.astype('int8'),
        'DISC_HOUR': hhmm // 100,               # NaN when the report has no discovery time
        'HOUR_MISSING': hhmm.isna().astype('int8'),
        'STACKED': stacked,
    })
    table = pd.concat([table, st], axis=1)
    for i in (1, 2, 3):
        table[f'LOG2_D{i}'] = np.log2(table[f'D{i}_KM'] + 1)   # one unit = one doubling of (distance + 1 km)
    table[f'LOG1P_STATIONS_{STATION_RADIUS_KM}KM'] = np.log1p(table[f'STATIONS_{STATION_RADIUS_KM}KM'])
    for c in ('STATE', 'REGION', 'OWNER', 'CAUSE'):
        table[c] = table[c].astype('category')
    return table


def load_feature_table(cache_path, root=None, rebuild=False):
    """Read the cached feature table, building it from the raw data the first time."""
    cache_path = Path(cache_path)
    if cache_path.is_file() and not rebuild:
        return pd.read_parquet(cache_path)
    db, stations_path = data_paths(root)
    table = build_table(load_fires(db), load_stations(stations_path))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cache_path, index=False)
    return table
