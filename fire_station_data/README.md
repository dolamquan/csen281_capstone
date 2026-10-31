# Fire and EMS Stations (HIFLD Open)

Second dataset for the capstone. The FPA-FOD wildfire database records where fires started, but not where firefighting resources are, so station locations come from here.

| | |
|---|---|
| Dataset | HIFLD Open: Fire and Emergency Medical Service (EMS) Stations |
| Original producer | U.S. Geological Survey (The National Map structures, feature code 74026) |
| Distributed by | DHS Homeland Infrastructure Foundation-Level Data (HIFLD) Open, archived on DataLumos (project 239161) |
| Download | https://www.datalumos.org/datalumos/project/239161 (mirror: https://portal.datarescueproject.org/datasets/hifld-open-fire-and-emergency-medical-service-ems-stations/) |
| Records | 52,057 point features (50 states, DC, PR, VI, GU, MP, AS) |
| Record load dates | 2010-2025, mostly 2015-2023 (a present-day snapshot, not a history) |
| CRS | WGS84 lon/lat (`urn:ogc:def:crs:OGC:1.3:CRS84`) |

## Layout

```
fire_station_data/
  hifld_fire_ems_stations_239161-V1.zip   original download (GeoJSON + shapefile + metadata)
  Data/
    fire-and-emergency-medical-service-ems-stations-geojson.geojson
    metadata.xml
```

The notebooks read `Data/*.geojson`. If it is missing, download the zip from the link above into this folder and extract the GeoJSON zip inside it into `Data/`.

## Fields used

`NAME`, `ADDRESS`, `CITY`, `STATE`, `ZIPCODE`, `LOADDATE`, `FCODE`, plus the point geometry (longitude, latitude).

## Known limitations

- Mostly municipal/county stations. Only a few hundred records are clearly wildland stations (CAL FIRE, USFS, BLM, state forestry), so distances on federal wildland are overstated.
- The producer notes that completeness varies by region and some rural locations are approximate.
- It is a current snapshot, so stations that opened or closed between 1992 and today are not tracked.
