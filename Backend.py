import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.ipac.irsa import Irsa
from astropy.utils.data import conf
from reproject import reproject_interp
import time

conf.remote_timeout = 120
os.makedirs("data", exist_ok=True)

TARGETS = {
    "wise0855": "08h55m10.831s -07d14m42.53s",
    "barnards_star": "17h57m48.498s +04d41m36.21s",
    "epsilon_indi_b": "22h04m10.520s -56d46m57.70s",
    "rw_aurigae": "05h07m49.540s +30d24m05.07s",
    "r_draconis": "16h32m40.230s +66d45m17.90s",
}

from astropy.nddata import Cutout2D
from scipy.ndimage import median_filter

def build_north_up_wcs(coord, shape_px, scale_arcsec=6.2):
    w = WCS(naxis=2)
    w.wcs.crpix = [shape_px / 2.0, shape_px / 2.0]
    w.wcs.cdelt = np.array([-scale_arcsec / 3600.0, scale_arcsec / 3600.0])
    w.wcs.crval = [coord.ra.deg, coord.dec.deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w

def fetch_and_process(coord_str, cutout_px=120, max_frames=5, max_spec=12):
    coord = SkyCoord(coord_str, frame="icrs")
    ra_deg, dec_deg = coord.ra.deg, coord.dec.deg
    df = None

    # 1. Primary Server: IRSA TAP Database (fast server-side cutouts)
    try:
        adql = f"""
        SELECT 'https://irsa.ipac.caltech.edu/' || a.uri || '?center={ra_deg},{dec_deg}d&size=0.25' AS access_url,
               p.time_bounds_lower AS t_min,
               p.energy_bandpassname,
               p.energy_bounds_lower AS em_min
        FROM spherex.artifact a
        JOIN spherex.plane p ON a.planeid = p.planeid
        WHERE 1 = CONTAINS(POINT('ICRS', {ra_deg}, {dec_deg}), p.poly)
        ORDER BY p.time_bounds_lower
        """
        res_tap = Irsa.query_tap(adql).to_table()
        if len(res_tap) > 0:
            df = res_tap.to_pandas()
            print("Fetched metadata via IRSA TAP server.")
    except Exception as e_tap:
        print(f"IRSA TAP busy ({e_tap}), switching to backup SIA server...")

    # 2. Backup Server: IRSA SIA Image Service
    if df is None or len(df) == 0:
        for attempt in range(2):
            try:
                res_sia = Irsa.query_sia(pos=(coord, 2 * u.arcsec), collection=['spherex_dr1', 'spherex_qr3'])
                if len(res_sia) > 0:
                    df = res_sia.to_pandas()
                    print("Fetched metadata via backup IRSA SIA server.")
                    break
            except Exception as e_sia:
                print(f"IRSA SIA attempt {attempt + 1}/2 failed: {e_sia}")
                time.sleep(2)

    if df is None or len(df) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    if 't_min' not in df.columns and 'time_bounds_lower' in df.columns:
        df['t_min'] = df['time_bounds_lower']
    if 't_min' in df.columns:
        df = df.sort_values(by='t_min')

    raw_em = pd.to_numeric(df.get('em_min', 1.5), errors='coerce').fillna(1.5)
    df['wave_um'] = np.where(raw_em < 0.01, raw_em * 1e6, raw_em)



    # Filter to same bandpass and pick well-separated epochs across the full time baseline
    valid_bands = df['energy_bandpassname'].dropna()
    df_band = df[df['energy_bandpassname'] == valid_bands.iloc[0]].copy() if len(valid_bands) > 0 else df.copy()
    
    # Group nearby dithers taken within 1 day so each epoch represents a distinct sky visit
    df_band['visit_day'] = (df_band['t_min'] // 5).astype(int)
    df_visits = df_band.drop_duplicates(subset=['visit_day']).sort_values(by='t_min')
    if len(df_visits) < 2:
        df_visits = df_band

    t_indices = np.linspace(0, len(df_visits) - 1, min(max_frames, len(df_visits)), dtype=int)
    sample_time_df = df_visits.iloc[t_indices]

    # Master North-Up WCS locked onto target coordinates
    target_wcs = build_north_up_wcs(coord, cutout_px, scale_arcsec=6.2)

    frames, dates, flags_summary = [], [], []
    for _, row in sample_time_df.iterrows():
        try:
            with fits.open(row['access_url'], memmap=False) as hdul:
                raw_wcs = WCS(hdul['IMAGE'].header, relax=True)
                clean_img = hdul['IMAGE'].data - hdul['ZODI'].data

                # Mask flagged bad pixels / cosmic rays before reprojection
                if 'FLAGS' in hdul:
                    bad_mask = hdul['FLAGS'].data > 0
                    flagged_pct = round(float(np.mean(bad_mask) * 100), 1)
                    clean_img = np.where(bad_mask, np.nan, clean_img)
                else:
                    flagged_pct = 0.0

                # Fill NaNs with local median so interpolation doesn't create black holes
                med_val = np.nanmedian(clean_img)
                clean_img = np.nan_to_num(clean_img, nan=med_val)

                # Reproject directly onto the locked North-Up target grid
                aligned, footprint = reproject_interp((clean_img, raw_wcs), target_wcs, shape_out=(cutout_px, cutout_px))
                aligned = np.nan_to_num(aligned, nan=np.nanmedian(aligned))

                # Skip frames where target fell off the edge of the detector (>25% empty)
                if np.mean(footprint > 0.5) < 0.75:
                    continue

                frames.append(aligned)
                dates.append(str(round(row['t_min'], 2)))
                flags_summary.append(f"{flagged_pct}% flagged")
        except Exception as e:
            print(f"Skipping time frame: {e}")

    # Part 2: Multi-wavelength stack on the exact same North-Up WCS grid
    df_wave = df.sort_values(by='wave_um')
    w_indices = np.linspace(0, len(df_wave) - 1, min(max_spec, len(df_wave)), dtype=int)
    sample_wave_df = df_wave.iloc[w_indices]

    spec_frames, spec_waves = [], []
    for _, row in sample_wave_df.iterrows():
        try:
            with fits.open(row['access_url'], memmap=False) as hdul:
                raw_wcs = WCS(hdul['IMAGE'].header, relax=True)
                clean_img = hdul['IMAGE'].data - hdul['ZODI'].data
                clean_img = np.nan_to_num(clean_img, nan=np.nanmedian(clean_img))
                aligned, footprint = reproject_interp((clean_img, raw_wcs), target_wcs, shape_out=(cutout_px, cutout_px))
                if np.mean(footprint > 0.5) < 0.75:
                    continue
                spec_frames.append(np.nan_to_num(aligned, nan=np.nanmedian(aligned)))
                spec_waves.append(float(row['wave_um']))
        except Exception:
            continue

    return np.array(frames), np.array(dates), np.array(flags_summary), np.array(spec_frames), np.array(spec_waves)


for name, coords in TARGETS.items():
    print(f"Processing {name}...")
    frames, dates, flags, spec_frames, spec_waves = fetch_and_process(coords)
    if len(frames) > 0:
        np.savez(f"data/{name}.npz", frames=frames, dates=dates, flags=flags, spec_frames=spec_frames, spec_waves=spec_waves)
        print(f"Saved data/{name}.npz ({len(frames)} time frames, {len(spec_frames)} spectral bands)")
