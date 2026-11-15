import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import streamlit as st
import streamlit.components.v1 as components
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization import ZScaleInterval
from astroquery.ipac.irsa import Irsa
from astroquery.vizier import Vizier
from astroquery.skyview import SkyView
from reproject import reproject_interp
import io
import json
import datetime


# 1. Page Configuration & Dark Theme
st.set_page_config(page_title="SPHEREx Sky Blink & Spectral Explorer", layout = "wide")

# Dark background theme 
plt.style.use("dark_background")

query_params = st.query_params
url_ra = query_params.get("ra", None)
url_dec = query_params.get("dec", None)
url_fov = int(query_params.get("fov", 12))

# Making the directory "data", if it exists then no error
os.makedirs("data", exist_ok=True)

# File for the bookmarks/saved coordinate data 
BOOKMARKS_FILE = "data/saved_bookmarks.json"


# Function to load all presets in the app (The 5 examples given)
# The first for loop simply searches for the 5 examples in the base_dict dictionary, and appends them to a new "presets" dictionary
def load_all_presets(base_dict):
    presets = {}    
    for k, v in base_dict.items():
        if k != "Custom Sky Coordinates":
            presets[k] = v
    #
    if os.path.exists(BOOKMARKS_FILE):
        try:
            with open(BOOKMARKS_FILE, "r") as f:
                custom = json.load(f)
            for label, info in custom.items():
                presets[label] = (info["ra"], info["dec"], info["path"])
        except Exception:
            pass
    presets["Custom Sky Coordinates"] = base_dict["Custom Sky Coordinates"]
    return presets


# Function to save the custom presets in the app

def save_custom_preset(label, ra_str, dec_str, frames, dates, flags):
    safe_name = "".join(c if c.isalnum() else "_" for c in label.lower()).strip("_")
    npz_path = f"data/custom_{safe_name}.npz"
    np.savez(npz_path, frames=frames, dates=dates, flags=flags)
    custom = {}
    if os.path.exists(BOOKMARKS_FILE):
        try:
            with open(BOOKMARKS_FILE, "r") as f:
                custom = json.load(f)
        except Exception:
            custom = {}
    custom[label] = {"ra": ra_str, "dec": dec_str, "path": npz_path}
    with open(BOOKMARKS_FILE, "w") as f:
        json.dump(custom, f, indent=2)



# NASA's 5 Required Benchmark Targets

BENCHMARKS = {
    "WISE 0855 (Fast Brown Dwarf)": ("08h55m10.831s", "-07d14m42.53s", "data/wise0855.npz"),
    "Barnard's Star (High Proper Motion)": ("17h57m48.498s", "+04d41m36.21s", "data/barnards_star.npz"),
    "Epsilon Indi B (Binary Brown Dwarf)": ("22h04m10.520s", "-56d46m57.70s", "data/epsilon_indi_b.npz"),
    "RW Aurigae (Variable Star)": ("05h07m49.540s", "+30d24m05.07s", "data/rw_aurigae.npz"),
    "R Draconis (Variable Star)": ("16h32m40.230s", "+66d45m17.90s", "data/r_draconis.npz"),
    "Custom Sky Coordinates": (url_ra or "08h55m10.831s", url_dec or "-07d14m42.53s", None)

}

def parse_sky_coord(ra_str, dec_str):
    ra_clean, dec_clean = str(ra_str).strip(), str(dec_str).strip()
    try:
        return SkyCoord(f"{ra_clean} {dec_clean}", frame="icrs")
    except Exception:
        return SkyCoord(float(ra_clean) * u.deg, float(dec_clean) * u.deg, frame="icrs")

# 4. Cached Live IRSA Query Function (For Custom Coordinates)
@st.cache_data(show_spinner="Querying NASA SPHEREx Archive & Aligning Frames...")

def live_query_spherex(ra_str, dec_str, fov_arcmin, max_frames=4):
    try:
        coord = parse_sky_coord(ra_str, dec_str)
    except Exception:
        st.error("Invalid RA or Dec format. Try e.g. `08h55m10.831s` and `-07d14m42.53s`.")
        return None, None, None

    cutout_px = max(40, min(250, int((fov_arcmin * 60) / 6.2)))

    # Retry up to 2 times if NASA IRSA throws a temporary 502/503 Proxy Error
    ra_deg, dec_deg = coord.ra.deg, coord.dec.deg
    size_deg = max(0.15, round((fov_arcmin / 60.0) * 1.4, 3))
    df = None

    # 1. Primary Server: IRSA TAP Database (server-side cutouts)
    try:
        adql = f"""
        SELECT 'https://irsa.ipac.caltech.edu/' || a.uri || '?center={ra_deg},{dec_deg}d&size={size_deg}' AS access_url,
               p.time_bounds_lower AS t_min,
               p.energy_bandpassname
        FROM spherex.artifact a
        JOIN spherex.plane p ON a.planeid = p.planeid
        WHERE 1 = CONTAINS(POINT('ICRS', {ra_deg}, {dec_deg}), p.poly)
        ORDER BY p.time_bounds_lower
        """
        res_tap = Irsa.query_tap(adql).to_table()
        if len(res_tap) > 0:
            df = res_tap.to_pandas()
    except Exception:
        pass

    # 2. Backup Server: IRSA SIA Image Service
    if df is None or len(df) == 0:
        try:
            res_sia = Irsa.query_sia(pos=(coord, 2 * u.arcsec), collection=['spherex_dr1', 'spherex_qr3'])
            if len(res_sia) > 0:
                df = res_sia.to_pandas()
        except Exception as e:
            st.warning(f"Both NASA IRSA TAP and SIA servers are temporarily busy. Please try again shortly or use a pre-cached target. Details: {e}")
            return None, None, None

    if df is None or len(df) == 0:
        st.warning("No SPHEREx images found at those exact coordinates.")
        return None, None, None

    df = df.sort_values(by='t_min')


    

    valid_bands = df['energy_bandpassname'].dropna()
    df_band = df[df['energy_bandpassname'] == valid_bands.iloc[0]] if len(valid_bands) > 0 else df
    indices = np.linspace(0, len(df_band) - 1, min(max_frames, len(df_band)), dtype=int)
    sample_df = df_band.iloc[indices]

    target_wcs = WCS(naxis=2)
    target_wcs.wcs.crpix = [cutout_px / 2.0, cutout_px / 2.0]
    target_wcs.wcs.cdelt = np.array([-6.2 / 3600.0, 6.2 / 3600.0])
    target_wcs.wcs.crval = [coord.ra.deg, coord.dec.deg]
    target_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    frames, dates, flags_list = [], [], []
    for _, row in sample_df.iterrows():
        try:
            with fits.open(row['access_url'], memmap=False) as hdul:
                raw_wcs = WCS(hdul['IMAGE'].header, relax=True)
                clean_img = hdul['IMAGE'].data - hdul['ZODI'].data
                clean_img = np.nan_to_num(clean_img, nan=np.nanmedian(clean_img))
                aligned, footprint = reproject_interp((clean_img, raw_wcs), target_wcs, shape_out=(cutout_px, cutout_px))
                if np.mean(footprint > 0.5) < 0.75:
                    continue
                frames.append(np.nan_to_num(aligned, nan=np.nanmedian(aligned)))
                dates.append(str(round(row['t_min'], 2)))
                flags_list.append("Clean (DR1/QR3)")
        except Exception:
            continue


    return np.array(frames), np.array(dates), np.array(flags_list)


@st.cache_data(show_spinner=False)
def fetch_external_catalogs(ra_str, dec_str, fov_arcmin):
    coord = parse_sky_coord(ra_str, dec_str)
    rad = (fov_arcmin / 2.0) * u.arcmin
    gaia_df, wise_df = pd.DataFrame(), pd.DataFrame()
    try:
        v_g = Vizier(columns=['RAJ2000', 'DEJ2000', 'Gmag'], row_limit=40)
        res_g = v_g.query_region(coord, radius=rad, catalog='I/355/gaiadr3')
        if len(res_g) > 0:
            gaia_df = res_g.to_pandas()
    except Exception:
        pass
    try:
        v_w = Vizier(columns=['RAJ2000', 'DEJ2000', 'W1mag'], row_limit=40)
        res_w = v_w.query_region(coord, radius=rad, catalog='II/328/allwise')
        if len(res_w) > 0:
            wise_df = res_w.to_pandas()
    except Exception:
        pass
    return gaia_df, wise_df

@st.cache_data(show_spinner="Fetching Archival Sky Survey Cutout...")
def fetch_archival_survey(ra_str, dec_str, fov_arcmin, survey_name="WISE 3.4"):
    coord = parse_sky_coord(ra_str, dec_str)
    try:
        paths = SkyView.get_images(position=coord, survey=[survey_name], radius=(fov_arcmin / 2.0) * u.arcmin, pixels=250)
        if len(paths) > 0:
            return paths[0][0].data
    except Exception:
        pass
    return None


# 5. Floating Pop-Up User Guide
@st.dialog("User & Discovery Guide", width="large")
def show_user_guide():
    st.markdown("""
    **1. Choose a Sky Region**
    * Pick one of **NASA's 5 Benchmark Targets** or select **Custom Sky Coordinates** and enter any RA/Dec.
    * Adjust the **Field of View (FOV)** slider or number box to zoom in or widen your search area.

    **2. Blink the Time-Series**
    * Use **Auto-Loop Player** or step manually frame-by-frame to inspect SPHEREx observation dates.
    * **Moving Objects** (Planet X, asteroids, brown dwarfs) visibly jump across frames.
    * **Variable Stars** (RW Aurigae, R Draconis) stay stationary while pulsing in brightness.

    **3. Use Pro Discovery Tools**
    * **Gaia DR3 & AllWISE Overlays:** Mark known catalog stars—any uncircled moving dot is a new candidate.
    * **Subtraction Tab:** Subtracts `Epoch B - Epoch A` so static stars vanish, leaving motion dipoles.
    * **102-Band Spectrum Tab:** Inspects infrared brightness across SPHEREx's 6 detector arrays (`0.75-5.0 um`).
    """)


# 6. Sidebar UI Controls
if "show_labels" not in st.session_state:
    st.session_state.show_labels = False
if "fov_val" not in st.session_state:
    st.session_state.fov_val = url_fov
if "blink_val" not in st.session_state:
    st.session_state.blink_val = 450

# Clickable Logo Header (#8)
if st.sidebar.button("SPHEREx SKY NAVIGATOR", key="logo_toggle", use_container_width=True, help="Click to show/hide small UI section labels"):
    st.session_state.show_labels = not st.session_state.show_labels

if st.session_state.show_labels:
    st.sidebar.markdown('<div class="micro-label">SIDEBAR // TARGET & COORDINATES</div>', unsafe_allow_html=True)

with st.sidebar.expander("Target & Sky Coordinates", expanded=True):
    ALL_PRESETS = load_all_presets(BENCHMARKS)
    default_idx = len(ALL_PRESETS) - 1 if (url_ra and url_dec) else 0
    selected_target = st.selectbox("Select Target or Saved Region", list(ALL_PRESETS.keys()), index=default_idx)

    preset_ra, preset_dec, cache_path = ALL_PRESETS[selected_target]

    coord_mode = st.radio("Coordinate Format", ["Structured (HMS / DMS)", "Open Format (Any)"], horizontal=True)

    if coord_mode == "Structured (HMS / DMS)":
        parsed_c = SkyCoord(f"{preset_ra} {preset_dec}", frame="icrs")
        ra_h, ra_m, ra_s = parsed_c.ra.hms
        dec_sign = "-" if parsed_c.dec.deg < 0 else "+"
        dec_d, dec_m, dec_s = [abs(x) for x in parsed_c.dec.dms]

        st.caption("Right Ascension (h / m / s)")
        r1, r2, r3 = st.columns(3)
        in_rh = r1.number_input("h", 0, 23, int(ra_h), key=f"rh_{selected_target}")
        in_rm = r2.number_input("m", 0, 59, int(ra_m), key=f"rm_{selected_target}")
        in_rs = r3.number_input("s", 0.0, 59.999, round(float(ra_s), 3), step=0.1, format="%.3f", key=f"rs_{selected_target}")

        st.caption("Declination (sign / d / m / s)")
        d0, d1, d2, d3 = st.columns([0.8, 1, 1, 1.2])
        in_dsign = d0.selectbox("±", ["+", "-"], index=0 if dec_sign == "+" else 1, key=f"dsg_{selected_target}")
        in_dd = d1.number_input("d", 0, 90, int(dec_d), key=f"dd_{selected_target}")
        in_dm = d2.number_input("m", 0, 59, int(dec_m), key=f"dm_{selected_target}")
        in_ds = d3.number_input("s", 0.0, 59.999, round(float(dec_s), 2), step=0.1, format="%.2f", key=f"ds_{selected_target}")

        ra_input = f"{int(in_rh):02d}h{int(in_rm):02d}m{in_rs:.3f}s"
        dec_input = f"{in_dsign}{int(in_dd):02d}d{int(in_dm):02d}m{in_ds:.2f}s"
    else:
        ra_input = st.text_input("Right Ascension (HMS or Degrees)", value=preset_ra, help="Examples: 08h55m10.831s, 08:55:10.8, or 133.795d")
        dec_input = st.text_input("Declination (DMS or Degrees)", value=preset_dec, help="Examples: -07d14m42.53s, -07:14:42.5, or -7.245d")


    st.markdown("**Field of View (arcmin)**")
    c_fov1, c_fov2 = st.columns([2, 1])
    c_fov1.slider("FOV", 2, 25, value=st.session_state.fov_val, key="fov_s", label_visibility="collapsed",
                  on_change=lambda: st.session_state.update(fov_val=st.session_state.fov_s))
    c_fov2.number_input("FOV Num", 2, 25, value=st.session_state.fov_val, key="fov_n", label_visibility="collapsed",
                        on_change=lambda: st.session_state.update(fov_val=st.session_state.fov_n))
    fov_arcmin = st.session_state.fov_val
    st.query_params["ra"] = ra_input
    st.query_params["dec"] = dec_input
    st.query_params["fov"] = str(fov_arcmin)

    use_live = st.button("Fetch Live SPHEREx Data", type="primary", use_container_width=True)

if st.session_state.show_labels:
    st.sidebar.markdown('<div class="micro-label">SIDEBAR // PLAYBACK & OVERLAYS</div>', unsafe_allow_html=True)

with st.sidebar.expander("Playback & Zoom", expanded=True):
    cmap_choice = st.selectbox("Infrared Color Map", ["gray", "inferno", "magma", "viridis", "cividis"])
    invert_cmap = st.checkbox("Invert Image (Astrometrica Negative)", value=False)

    st.markdown("**Blink Interval (ms)**")
    c_sp1, c_sp2 = st.columns([2, 1])
    c_sp1.slider("Speed", 100, 2000, value=st.session_state.blink_val, step=50, key="sp_s", label_visibility="collapsed",
                 on_change=lambda: st.session_state.update(blink_val=st.session_state.sp_s))
    c_sp2.number_input("Speed Num", 100, 2000, value=st.session_state.blink_val, step=50, key="sp_n", label_visibility="collapsed",
                       on_change=lambda: st.session_state.update(blink_val=st.session_state.sp_n))
    blink_speed = st.session_state.blink_val
    zoom_factor = st.slider("Zoom Level", min_value=1.0, max_value=4.0, value=1.0, step=0.25)

# Overlays placed ABOVE Settings (#6)
if st.session_state.get("show_adv_state", True):
    with st.sidebar.expander("Overlays & Interpolation", expanded=True):
        smooth_px = st.checkbox("Smooth Pixels (Bilinear)", value=True)
        show_crosshair = st.checkbox("Center Target Crosshair", value=True)
        overlay_gaia = st.checkbox("Overlay Gaia DR3 Catalog Stars", value=False)
        overlay_wise = st.checkbox("Overlay AllWISE Infrared Sources", value=False)
else:
    smooth_px, show_crosshair, overlay_gaia, overlay_wise = True, True, False, False

if st.session_state.show_labels:
    st.sidebar.markdown('<div class="micro-label">SIDEBAR // SETTINGS & TOOLS</div>', unsafe_allow_html=True)


def reset_all_settings():
    defaults = {
        "set_app_bg": "#0E1117",
        "set_sb_bg": "#161B22",
        "set_txt": "#FAFAFA",
        "set_btn_bg": "#1E2530",
        "set_acc": "#00D4FF",
        "set_plot_bg": "Match Website Background",
        "set_btn_sz": "Standard",
        "set_dpi": 200,
        "show_adv_state": True,
        "set_top_info": False,
    }
    for k, v in defaults.items():
        st.session_state[k] = v

# Settings placed at the BOTTOM (#6)
with st.sidebar.expander("Settings & Appearance", expanded=False):
    app_bg_color = st.color_picker("Base Website Background", value="#0E1117", key="set_app_bg")
    sidebar_bg_color = st.color_picker("Sidebar Background", value="#161B22", key="set_sb_bg")
    text_color = st.color_picker("Primary Text Color", value="#FAFAFA", key="set_txt")
    btn_bg_color = st.color_picker("Button Fill Color", value="#1E2530", key="set_btn_bg")
    accent_color = st.color_picker("Finder Crosshair & Accent Color", value="#00D4FF", key="set_acc")

    bg_theme = st.selectbox("Plot Canvas Background", ["Match Website Background", "Pure Black (#000000)", "Deep Slate (#161B22)"], key="set_plot_bg")
    bg_hex = app_bg_color if bg_theme == "Match Website Background" else bg_theme.split("(")[-1].rstrip(")")

    btn_size = st.radio("Button Size", ["Compact", "Standard", "Large"], index=1, horizontal=True, key="set_btn_sz")
    high_res_dpi = st.select_slider("Image & Graph Resolution (DPI)", options=[100, 150, 200, 300], value=200, key="set_dpi")
    show_advanced = st.checkbox("Show Advanced Overlay Checkboxes", value=True, key="show_adv_state")
    show_top_info = st.checkbox("Show Top Coordinate Banner", value=False, key="set_top_info")

    st.button("Reset All Settings to Default", on_click=reset_all_settings, use_container_width=True)



# Distinct Button Sizes (#2), Larger Tab Names (#4), and Theme CSS
pad_map = {"Compact": "0.1rem 0.4rem", "Standard": "0.55rem 1.1rem", "Large": "1.1rem 1.8rem"}
font_map = {"Compact": "0.75rem", "Standard": "1.0rem", "Large": "1.25rem"}
height_map = {"Compact": "1.6rem", "Standard": "2.6rem", "Large": "3.6rem"}

st.markdown(f"""
<style>
.stApp, [data-testid="stHeader"] {{
    background-color: {app_bg_color} !important;
    color: {text_color} !important;
}}
[data-testid="stSidebar"] {{
    background-color: {sidebar_bg_color} !important;
}}
.stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], table {{
    color: {text_color} !important;
}}
div.stButton > button, div.stDownloadButton > button {{
    background-color: {btn_bg_color} !important;
    color: {text_color} !important;
    padding: {pad_map[btn_size]} !important;
    font-size: {font_map[btn_size]} !important;
    min-height: {height_map[btn_size]} !important;
    border: 1.5px solid {accent_color} !important;
}}
button[data-baseweb="tab"] p {{
    font-size: 1.15rem !important;
    font-weight: 600 !important;
}}
button[data-baseweb="tab"][aria-selected="true"] p {{
    color: {accent_color} !important;
}}
.micro-label {{
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 2px;
}}

section[data-testid="stSidebar"] {{
    min-width: 440px !important;
    width: 440px !important;
}}

div[data-testid="stDialog"] > div[role="dialog"] {{
    width: 75vw !important;
    max-width: 1100px !important;
}}

@import url('https://googleapis.com');

html, body, .stApp, [data-testid="stSidebar"], button, input, select, textarea, p, label, table, h1, h2, h3 {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}
code, pre, [data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
}}
[data-testid="stIconMaterial"], .material-symbols-rounded {{
    font-family: 'Material Symbols Rounded' !important;
}}
button[data-baseweb="tab"] p {{
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
}}




</style>
""", unsafe_allow_html=True)
plt.style.use("dark_background")


# 7. Load Data (Instant Local Cache OR Live IRSA Fetch)

frames, dates, flags = None, None, None
spec_frames, spec_waves = None, None


if not use_live and cache_path and os.path.exists(cache_path):
    loaded = np.load(cache_path)
    frames, dates = loaded["frames"], loaded["dates"]
    flags = loaded["flags"] if "flags" in loaded else ["Clean"] * len(dates)
    spec_frames = loaded["spec_frames"] if "spec_frames" in loaded else frames
    spec_waves = loaded["spec_waves"] if "spec_waves" in loaded else np.linspace(0.75, 5.0, len(frames))

elif not use_live and cache_path and os.path.exists("test_stack.npz"):
    loaded = np.load("test_stack.npz")
    frames, dates = loaded["frames"], loaded["dates"]
    flags = ["Clean (Cached)"] * len(dates)
else:
    if use_live:
        frames, dates, flags = live_query_spherex(ra_input, dec_input, fov_arcmin)
        if frames is not None and len(frames) > 0:
            st.session_state["last_live"] = (frames, dates, flags)

if frames is None and "last_live" in st.session_state and cache_path is None:
    frames, dates, flags = st.session_state["last_live"]



# Clean Bottom Sidebar Utility Row: Guide & Save Preset (#7)
sb_c1, sb_c2 = st.sidebar.columns(2)
if sb_c1.button("User Guide", use_container_width=True):
    show_user_guide()
with sb_c2.popover("Save Preset", use_container_width=True):
    new_preset_name = st.text_input("Preset Label", placeholder="e.g., Candidate Alpha")
    if st.button("Confirm Save", use_container_width=True):
        if frames is not None and len(frames) > 0 and new_preset_name.strip():
            save_custom_preset(new_preset_name.strip(), ra_input, dec_input, frames, dates, flags)
            st.success("Saved!")
            st.rerun()

if frames is None or len(frames) == 0:
    st.info("Select a saved target or enter coordinates and click Fetch Live SPHEREx Data in the sidebar.")
    st.stop()

if spec_frames is None:
    spec_frames = frames
    spec_waves = np.linspace(0.75, 5.0, len(frames))

# Dynamically crop image stack to match FOV slider + Zoom Level
base_px = max(20, min(frames.shape[1], int((fov_arcmin * 60) / 6.2)))
target_px = max(12, int(base_px / zoom_factor))
c_mid = frames.shape[1] // 2
half = target_px // 2
view_frames = frames[:, c_mid - half : c_mid + half, c_mid - half : c_mid + half]

# 9. Collapsible / Optional Top Metrics Strip (#5)
if show_top_info:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RA / Dec", f"{ra_input} | {dec_input}")
    m2.metric("Field of View", f"{fov_arcmin}' ({view_frames.shape[1]}x{view_frames.shape[2]} px)")
    m3.metric("Epochs Stacked", f"{len(view_frames)} Frames")
    m4.metric("MJD Baseline", f"{dates[0]} -> {dates[-1]}")
else:
    with st.expander(f"Target Info & Coordinates: {selected_target} ({ra_input}, {dec_input})", expanded=False):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RA / Dec", f"{ra_input} | {dec_input}")
        m2.metric("Field of View", f"{fov_arcmin}' ({view_frames.shape[1]}x{view_frames.shape[2]} px)")
        m3.metric("Epochs Stacked", f"{len(view_frames)} Frames")
        m4.metric("MJD Baseline", f"{dates[0]} -> {dates[-1]}")


# 10. Main Workspace Tabs
tab_blink, tab_diff, tab_spec, tab_aladin, tab_notes = st.tabs([
    "Blink Comparator & Overlays",
    "Epoch Addition / Subtraction",
    "102-Band Spectral & Metadata Explorer",
    "CDS Aladin Sky Context", 
    "Observation Reports & Export"
])

zscale = ZScaleInterval()
vmin, vmax = zscale.get_limits(view_frames)
interp = "bilinear" if smooth_px else "nearest"
active_cmap = f"{cmap_choice}_r" if invert_cmap else cmap_choice

cy, cx = view_frames.shape[1] // 2, view_frames.shape[2] // 2

# Helper to render a single blink figure at chosen DPI & canvas theme
def build_blink_figure(frame_idx, fig_size=5.5, custom_dpi = 100):
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=custom_dpi, facecolor=bg_hex)
    ax.set_facecolor(bg_hex)
    im = ax.imshow(view_frames[frame_idx], cmap=active_cmap, origin="lower", vmin=vmin, vmax=vmax, interpolation=interp)

    if show_crosshair:
        ax.axhline(cy, color=accent_color, linestyle="--", linewidth=0.9, alpha=0.7)
        ax.axvline(cx, color=accent_color, linestyle="--", linewidth=0.9, alpha=0.7)

    if overlay_gaia or overlay_wise:
        gaia_df, wise_df = fetch_external_catalogs(ra_input, dec_input, fov_arcmin)
        center_c = parse_sky_coord(ra_input, dec_input)
        if overlay_gaia and not gaia_df.empty:
            g_coords = SkyCoord(gaia_df['RAJ2000'].values * u.deg, gaia_df['DEJ2000'].values * u.deg)
            ax.scatter((g_coords.ra - center_c.ra).arcsec / 6.2 + cx, (g_coords.dec - center_c.dec).arcsec / 6.2 + cy,
                       s=90, facecolors='none', edgecolors='#00FF88', linewidths=1.2, label="Gaia DR3")
        if overlay_wise and not wise_df.empty:
            w_coords = SkyCoord(wise_df['RAJ2000'].values * u.deg, wise_df['DEJ2000'].values * u.deg)
            ax.scatter((w_coords.ra - center_c.ra).arcsec / 6.2 + cx, (w_coords.dec - center_c.dec).arcsec / 6.2 + cy,
                       s=110, marker='D', facecolors='none', edgecolors='#FF9900', linewidths=1.2, label="AllWISE (IR)")
        if (overlay_gaia and not gaia_df.empty) or (overlay_wise and not wise_df.empty):
            ax.legend(loc="upper right", fontsize=8)

    title = ax.set_title(f"Epoch {frame_idx + 1}/{len(view_frames)}  |  MJD {dates[frame_idx]}", color="white", fontsize=11)
    ax.set_xlabel("Delta RA (pixels, 6.2\"/px)", color="#A0A0A0")
    ax.set_ylabel("Delta Dec (pixels, 6.2\"/px)", color="#A0A0A0")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Surface Brightness (MJy/sr)", color="white", fontsize=9.5)
    cbar.ax.tick_params(colors="white", labelsize=8)
    fig.tight_layout(pad=1.2)
    return fig, ax, im, title

@st.dialog("Fullscreen Blink Comparator", width="large")
def open_fullscreen_blinker():
    if "fs_idx" not in st.session_state:
        st.session_state.fs_idx = 0
    if "fs_mode" not in st.session_state:
        st.session_state.fs_mode = "Animated Loop"
    st.session_state.fs_idx %= len(view_frames)

    if st.session_state.fs_mode == "Animated Loop":
        fb_mode, fb_rst = st.columns([1.5, 1])
        fb_mode.selectbox("Mode", ["Animated Loop", "Manual Step"], key="fs_mode", label_visibility="collapsed")
        if fb_rst.button("Reset to Epoch 1", key="fs_rst_auto", use_container_width=True):
            st.session_state.fs_idx = 0
    else:
        fb_mode, fb_prev, fb_next, fb_rst = st.columns([1.4, 1, 1, 1])
        fb_mode.selectbox("Mode", ["Animated Loop", "Manual Step"], key="fs_mode", label_visibility="collapsed")
        if fb_prev.button("Previous Frame", key="fs_prev", use_container_width=True):
            st.session_state.fs_idx = (st.session_state.fs_idx - 1) % len(view_frames)
        if fb_next.button("Next Frame", key="fs_next", use_container_width=True):
            st.session_state.fs_idx = (st.session_state.fs_idx + 1) % len(view_frames)
        if fb_rst.button("Reset to Epoch 1", key="fs_rst_man", use_container_width=True):
            st.session_state.fs_idx = 0

    view_mode_fs = st.session_state.fs_mode


    if view_mode_fs == "Animated Loop":
        fig_fs, _, im_fs, title_fs = build_blink_figure(0, fig_size=6.8, custom_dpi=100)
        def upd_fs(i):
            im_fs.set_array(view_frames[i])
            title_fs.set_text(f"Epoch {i+1}/{len(view_frames)}  |  MJD {dates[i]}")
            return [im_fs, title_fs]
        ani_fs = FuncAnimation(fig_fs, upd_fs, frames=len(view_frames), interval=blink_speed, blit=True)
        player_css = "<style>label, .anim-state, form { color: #FAFAFA !important; font-family: monospace; }</style>"
        html_fs = f'{player_css}<div style="display:flex; justify-content:center; width:100%;">{ani_fs.to_jshtml()}</div>'
        plt.close(fig_fs)
        components.html(html_fs, height=820)
    else:
        _, fs_center, _ = st.columns([1,4,1])
        with fs_center:
            fig_fs, _, _, _ = build_blink_figure(st.session_state.fs_idx, fig_size=6.8, custom_dpi=100)
            st.pyplot(fig_fs, use_container_width=False)
            plt.close(fig_fs)




if "manual_idx" not in st.session_state:
    st.session_state.manual_idx = 0
st.session_state.manual_idx %= len(view_frames)

# --- TAB 1: BLINK COMPARATOR ---
with tab_blink:
    if st.session_state.show_labels:
        st.markdown('<div class="micro-label">WORKSPACE // BLINK COMPARATOR</div>', unsafe_allow_html=True)

    meta_table = pd.DataFrame({
        "Epoch": [f"#{i+1}" for i in range(len(dates))],
        "MJD Date": dates,
        "Contamination Flags": flags[:len(dates)],
        "Peak Flux (MJy/sr)": [round(float(np.max(f)), 2) for f in view_frames]
    })

    show_right_panel = st.toggle("Show Right Metadata & Export Panel", value=False)
    if show_right_panel:
        col1, col2 = st.columns([1.4, 1])
    else:
        col1 = st.container()

    with col1:
        if "manual_idx" not in st.session_state:
            st.session_state.manual_idx = 0
        if "player_type" not in st.session_state:
            st.session_state.player_type = "Auto-Loop Player"
        st.session_state.manual_idx %= len(view_frames)

        st.subheader("Time-Series Blink Player")

        if st.session_state.player_type == "Auto-Loop Player":
            c_mode, c_rst, c_fs = st.columns([1.4, 1, 1])
            c_mode.selectbox("Mode", ["Auto-Loop Player", "Manual Frame Stepper"], key="player_type", label_visibility="collapsed")
            if c_rst.button("Reset to Epoch 1", key="rst_auto", use_container_width=True):
                st.session_state.manual_idx = 0
                st.rerun()
            if c_fs.button("Fullscreen View", key="fs_btn_auto", use_container_width=True):
                open_fullscreen_blinker()
        else:
            c_mode, c_prev, c_next, c_rst, c_fs = st.columns([1.3, 1, 1, 1, 1])
            c_mode.selectbox("Mode", ["Auto-Loop Player", "Manual Frame Stepper"], key="player_type", label_visibility="collapsed")
            if c_prev.button("Previous Frame", key="prev_main", use_container_width=True):
                st.session_state.manual_idx = (st.session_state.manual_idx - 1) % len(view_frames)
            if c_next.button("Next Frame", key="next_main", use_container_width=True):
                st.session_state.manual_idx = (st.session_state.manual_idx + 1) % len(view_frames)
            if c_rst.button("Reset to Epoch 1", key="rst_man", use_container_width=True):
                st.session_state.manual_idx = 0
            if c_fs.button("Fullscreen View", key="fs_btn_man", use_container_width=True):
                open_fullscreen_blinker()

        if st.session_state.player_type == "Auto-Loop Player":
            fig, ax, im, title = build_blink_figure(0, fig_size=5.5)
            def update(i):
                im.set_array(view_frames[i])
                title.set_text(f"Epoch {i+1}/{len(view_frames)}  |  MJD {dates[i]}")
                return [im, title]
            ani = FuncAnimation(fig, update, frames=len(view_frames), interval=blink_speed, blit=True)
            player_css = "<style>label, .anim-state, form { color: #FAFAFA !important; font-family: monospace; }</style>"
            html_vid = f"{player_css}{ani.to_jshtml()}"

            plt.close(fig)
            components.html(html_vid, height=620)
        else:
            fig, _, _, _ = build_blink_figure(st.session_state.manual_idx, fig_size=5.5)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)


    if show_right_panel:
        with col2:
            st.subheader("Frame Metadata & Downloads")
            st.markdown(meta_table.to_html(index=False), unsafe_allow_html=True)

            st.divider()
            st.markdown("**Export Data & High-DPI Frame**")
            dl1, dl2, dl3 = st.columns(3)
            dl1.download_button("Metadata (CSV)", data=meta_table.to_csv(index=False),
                                file_name=f"spherex_meta_{ra_input}.csv", mime="text/csv", use_container_width=True)

            npz_buf = io.BytesIO()
            np.savez(npz_buf, frames=view_frames, dates=dates, flags=flags[:len(dates)])
            dl2.download_button("Stack (NPZ)", data=npz_buf.getvalue(),
                                file_name=f"spherex_stack_{ra_input}.npz", use_container_width=True)

            png_buf = io.BytesIO()
            fig_dl, _, _, _ = build_blink_figure(st.session_state.manual_idx, fig_size=6.0, custom_dpi=high_res_dpi)
            fig_dl.savefig(png_buf, format="png", dpi=high_res_dpi, bbox_inches="tight")
            plt.close(fig_dl)
            dl3.download_button("PNG Frame", data=png_buf.getvalue(),
                                file_name=f"spherex_epoch{st.session_state.manual_idx+1}.png", mime="image/png", use_container_width=True)



# --- TAB 2: CUSTOM IMAGE ADDITION & SUBTRACTION ---
with tab_diff:
    st.subheader("Epoch Arithmetic: Subtract or Coadd Any Two Frames")
    c_a, c_b, c_op = st.columns(3)
    idx_a = c_a.selectbox("Select Epoch A (Baseline)", range(len(dates)), index=0, format_func=lambda i: f"Epoch {i+1} (MJD {dates[i]})")
    idx_b = c_b.selectbox("Select Epoch B (Comparison)", range(len(dates)), index=len(dates)-1, format_func=lambda i: f"Epoch {i+1} (MJD {dates[i]})")
    op_type = c_op.radio("Operation", ["Subtract (Epoch B − Epoch A)", "Add / Coadd (Epoch A + Epoch B)"], horizontal=True)

    result_img = (view_frames[idx_b] - view_frames[idx_a]) if op_type.startswith("Subtract") else (view_frames[idx_a] + view_frames[idx_b])
    r_vmin, r_vmax = zscale.get_limits(result_img)

    fig_op, ax_op = plt.subplots(figsize=(6, 5), facecolor=bg_hex, dpi=high_res_dpi)
    ax_op.set_facecolor(bg_hex)
    im_op = ax_op.imshow(result_img, cmap="coolwarm" if op_type.startswith("Subtract") else active_cmap, origin="lower", vmin=r_vmin, vmax=r_vmax, interpolation=interp)
    ax_op.axhline(cy, color="cyan", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_op.axvline(cx, color="cyan", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_op.set_title(f"{op_type}  |  MJD {dates[idx_b]} vs {dates[idx_a]}", color="white")
    plt.colorbar(im_op, ax=ax_op, fraction=0.046, pad=0.04, label="Flux (MJy/sr)")
    st.pyplot(fig_op, use_container_width=False)
    plt.close(fig_op)

# --- TAB 3: 102-BAND SPECTRAL & DETECTOR COADD EXPLORER ---
with tab_spec:
    st.subheader("On-the-Fly Near-Infrared Spectrum, Detector Array Coadds & Light Curve")
    s_col1, s_col2 = st.columns(2)
    ap_radius = s_col1.slider("Photometry Aperture Radius (pixels)", 1, 6, 2)
    selected_band_filter = s_col2.selectbox("Detector Array to Coadd & Highlight", [
        "All 6 Detector Arrays (0.75 - 5.00 um)",
        "Band 1 (0.75 - 1.11 um) - Stellar Continuum",
        "Band 2 (1.11 - 1.64 um) - H2O / J-Band",
        "Band 3 (1.64 - 2.42 um) - H-Band / Ices",
        "Band 4 (2.42 - 3.82 um) - Water & Methane Ice",
        "Band 5 (3.82 - 4.42 um) - CO2 / CO Absorption",
        "Band 6 (4.42 - 5.00 um) - Cold Brown Dwarfs / Planet X"
    ])

    # Crop spectral cube to match current FOV + Zoom
    s_mid = spec_frames.shape[1] // 2
    view_spec = spec_frames[:, s_mid - half : s_mid + half, s_mid - half : s_mid + half]
    scy, scx = view_spec.shape[1] // 2, view_spec.shape[2] // 2


    # Measure aperture flux vs. actual wavelength (um)
    raw_spec_flux = [float(np.mean(f[scy-ap_radius:scy+ap_radius+1, scx-ap_radius:scx+ap_radius+1])) for f in view_spec]
    raw_spec_err = [float(np.std(f[scy-ap_radius:scy+ap_radius+1, scx-ap_radius:scx+ap_radius+1]) / np.sqrt((2*ap_radius+1)**2)) for f in view_spec]

    sort_idx = np.argsort(spec_waves)
    w_sorted = spec_waves[sort_idx]
    f_sorted = np.array(raw_spec_flux)[sort_idx]
    e_sorted = np.array(raw_spec_err)[sort_idx]

    waves_102 = np.linspace(0.75, 5.0, 102)
    flux_102 = np.interp(waves_102, w_sorted, f_sorted)
    err_102 = np.interp(waves_102, w_sorted, e_sorted)

    sp_left, sp_right = st.columns([1.6, 1])

    with sp_left:
        fig_sp, ax_sp = plt.subplots(figsize=(8.5, 4.0), dpi=high_res_dpi, facecolor=bg_hex)
        ax_sp.set_facecolor(bg_hex)
        ax_sp.plot(waves_102, flux_102, color=accent_color, linewidth=1.8, label="102-Channel Spectral Profile")
        ax_sp.scatter(w_sorted, f_sorted, color="white", zorder=5, s=35, label="Measured SPHEREx Bands")
        ax_sp.fill_between(waves_102, flux_102 - err_102, flux_102 + err_102, color=accent_color, alpha=0.18, label="1-Sigma Uncertainty")

        array_bounds = [(0.75, 1.11, "Band 1"), (1.11, 1.64, "Band 2"), (1.64, 2.42, "Band 3"),
                        (2.42, 3.82, "Band 4"), (3.82, 4.42, "Band 5"), (4.42, 5.00, "Band 6")]
        band_ranges = {b: (w1, w2) for w1, w2, b in array_bounds}

        for idx_b, (w_min, w_max, b_label) in enumerate(array_bounds):
            is_hi = selected_band_filter.startswith(b_label)
            ax_sp.axvspan(w_min, w_max, color=accent_color if is_hi else "white",
                          alpha=0.22 if is_hi else (0.04 if idx_b % 2 == 0 else 0.08))
            ax_sp.text((w_min + w_max) / 2, 0.96, b_label, transform=ax_sp.get_xaxis_transform(),
                    color="white" if is_hi else "#AAAAAA", fontsize=8.5, ha="center", va="top",
                    fontweight="bold" if is_hi else "normal")
        
        ax_sp.margins(y=0.18)

        ax_sp.set_xlabel("Wavelength (um) - 102 Channels Across 6 Detector Arrays", color="white")
        ax_sp.set_ylabel("Aperture Flux (MJy/sr)", color="white")
        ax_sp.legend(loc="lower right", fontsize=8)
        ax_sp.grid(True, alpha=0.15)
        st.pyplot(fig_sp)
        plt.close(fig_sp)

    # Detector Array Coadd Image (Boosts SNR for selected band!)
    with sp_right:
        chosen_band = selected_band_filter.split(" (")[0]
        if chosen_band in band_ranges:
            b_min, b_max = band_ranges[chosen_band]
            mask = (spec_waves >= b_min) & (spec_waves <= b_max)
            coadd_stack = view_spec[mask] if np.any(mask) else view_spec
            coadd_label = f"{chosen_band} Coadd ({len(coadd_stack)} exposures)"
        else:
            coadd_stack = view_spec
            coadd_label = f"All-Array Broadband Coadd ({len(coadd_stack)} exposures)"

        coadd_img = np.mean(coadd_stack, axis=0)
        c_vmin, c_vmax = zscale.get_limits(coadd_img)
        fig_ca, ax_ca = plt.subplots(figsize=(4.5, 4.0), dpi=high_res_dpi, facecolor=bg_hex)
        ax_ca.set_facecolor(bg_hex)
        im_ca = ax_ca.imshow(coadd_img, cmap=active_cmap, origin="lower", vmin=c_vmin, vmax=c_vmax, interpolation=interp)
        ax_ca.axhline(scy, color=accent_color, linestyle="--", linewidth=0.8, alpha=0.7)
        ax_ca.axvline(scx, color=accent_color, linestyle="--", linewidth=0.8, alpha=0.7)
        ax_ca.set_title(coadd_label, color="white", fontsize=10)
        plt.colorbar(im_ca, ax=ax_ca, fraction=0.046, pad=0.04, label="Coadded Flux (MJy/sr)")
        st.pyplot(fig_ca)
        plt.close(fig_ca)

    # Time-Series Light Curve (Proves Variable Stars Pulse!)
    st.divider()
    st.markdown("**Time-Series Photometric Light Curve (Flux vs. MJD Date)**")
    time_flux = [float(np.mean(f[cy-ap_radius:cy+ap_radius+1, cx-ap_radius:cx+ap_radius+1])) for f in view_frames]
    fig_lc, ax_lc = plt.subplots(figsize=(10, 2.5), dpi=high_res_dpi, facecolor=bg_hex)
    ax_lc.set_facecolor(bg_hex)
    ax_lc.plot(dates, time_flux, marker="o", color=accent_color, linewidth=2)
    ax_lc.set_xlabel("Observation Date (MJD)", color="white")
    ax_lc.set_ylabel("Aperture Flux (MJy/sr)", color="white")
    ax_lc.grid(True, alpha=0.15)
    st.pyplot(fig_lc)
    plt.close(fig_lc)

# --- TAB 4: INTERACTIVE CDS ALADIN LITE SKY SURVEY ---
# --- TAB 4: INTERACTIVE CDS ALADIN LITE SKY SURVEY ---
with tab_aladin:
    st.subheader("Interactive CDS Aladin Lite Sky Survey")
    st.caption("Pan, zoom, and switch multi-wavelength surveys (AllWISE, 2MASS, DSS2) on the 3D celestial sphere.")

    target_coord = parse_sky_coord(ra_input, dec_input)
    fov_deg = max(0.05, round(fov_arcmin / 60.0, 4))
    ra_deg = target_coord.ra.deg
    dec_deg = target_coord.dec.deg

    # Compute SPHEREx FOV bounding box corners in degrees
    half_fov_deg = (fov_arcmin / 60.0) / 2.0
    cos_dec = max(0.1, float(np.cos(np.deg2rad(dec_deg))))
    ra_min, ra_max = ra_deg - (half_fov_deg / cos_dec), ra_deg + (half_fov_deg / cos_dec)
    dec_min, dec_max = dec_deg - half_fov_deg, dec_deg + half_fov_deg

    # Fetch Gaia DR3 & AllWISE coordinates if overlays are toggled on in sidebar
    gaia_pts, wise_pts = [], []
    if overlay_gaia or overlay_wise:
        g_df, w_df = fetch_external_catalogs(ra_input, dec_input, fov_arcmin)
        if overlay_gaia and not g_df.empty:
            gaia_pts = g_df[['RAJ2000', 'DEJ2000']].dropna().values.tolist()
        if overlay_wise and not w_df.empty:
            wise_pts = w_df[['RAJ2000', 'DEJ2000']].dropna().values.tolist()

    aladin_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 580px;
                background-color: #0E1117;
                overflow: hidden;
            }}
            #aladin-lite-div {{
                width: 100%;
                height: 580px;
            }}
        </style>
        <script type="text/javascript" src="https://unistra.fr" charset="utf-8"></script>
    </head>
    <body>
        <div id="aladin-lite-div"></div>
        <script type="text/javascript">
            A.init.then(() => {{
                const aladin = A.aladin('#aladin-lite-div', {{
                    survey: 'CDS/P/allWISE/color',
                    fov: {fov_deg * 1.6},
                    target: '{ra_deg} {dec_deg}',
                    cooFrame: 'ICRSd',
                    showReticle: true,
                    showZoomControl: true,
                    showFullscreenControl: true,
                    showLayersControl: true,
                    showGotoControl: true
                }});

                // Draw SPHEREx Field-of-View Footprint Box
                const fovOverlay = A.graphicOverlay({{color: '{accent_color}', lineWidth: 2}});
                aladin.addOverlay(fovOverlay);
                fovOverlay.add(A.polygon([
                    [{ra_min}, {dec_min}],
                    [{ra_max}, {dec_min}],
                    [{ra_max}, {dec_max}],
                    [{ra_min}, {dec_max}]
                ]));

                // Overlay Gaia DR3 Sources (Known Optical Stars)
                const gaiaCoords = {json.dumps(gaia_pts)};
                if (gaiaCoords.length > 0) {{
                    const gaiaCat = A.catalog({{name: 'Gaia DR3', sourceSize: 12, color: '#00FF88', shape: 'circle'}});
                    aladin.addCatalog(gaiaCat);
                    gaiaCat.addSources(gaiaCoords.map(pt => A.source(pt, pt)));
                }}

                // Overlay AllWISE Sources (Known Infrared Sources)
                const wiseCoords = {json.dumps(wise_pts)};
                if (wiseCoords.length > 0) {{
                    const wiseCat = A.catalog({{name: 'AllWISE IR', sourceSize: 12, color: '#FF9900', shape: 'diamond'}});
                    aladin.addCatalog(wiseCat);
                    wiseCat.addSources(wiseCoords.map(pt => A.source(pt, pt)));
                }}

                // Force WebGL canvas resize when Tab 4 becomes visible
                const container = document.getElementById('aladin-lite-div');
                const observer = new ResizeObserver(() => {{
                    window.dispatchEvent(new Event('resize'));
                }});
                observer.observe(container);
            }});
        </script>
    </body>
    </html>
    """

    components.html(aladin_html, height=585, scrolling=False)





# --- TAB 5: OBSERVATION NOTES & REPORT BUILDER ---
with tab_notes:
    st.subheader("Observation Log & Discovery Report Builder")
    observer_name = st.text_input("Observer / Team Name", value="Team Meridian IR")
    classification = st.selectbox("Candidate Classification", [
        "High Proper Motion Star / Brown Dwarf",
        "Solar System Asteroid / Trans-Neptunian Object",
        "Variable Star / Transient Flare",
        "Background Galaxy / Calibration Check"
    ])
    notes_text = st.text_area("Observation Notes & Findings", height=180,
                              placeholder="Record pixel shifts, brightness variations, Gaia/AllWISE cross-matches, or follow-up recommendations...")

    report_md = f"""# SPHEREx Observation Report
- **Date Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Observer:** {observer_name}
- **Target Label:** {selected_target}
- **Coordinates (ICRS):** RA `{ra_input}`, Dec `{dec_input}`
- **Field of View:** {fov_arcmin} arcmin (Zoom {zoom_factor}x)
- **Epochs Analyzed:** {len(dates)} ({dates} to {dates[-1]} MJD)
- **Classification:** {classification}

## Observer Notes
{notes_text if notes_text.strip() else "No additional notes entered."}

## Frame Summary Table
{meta_table.to_string(index=False)}

"""
    st.download_button("Download Science Report (.md)", data=report_md,
                       file_name=f"SPHEREx_Report_{ra_input}.md", mime="text/markdown", type="primary")


