import streamlit as st
import numpy as np
import plotly.graph_objects as go
from numba import njit
from scipy.signal.windows import chebwin

st.set_page_config(layout="wide", page_title="Multibeam Echosounder Mills Cross Simulator")

st.title("Multibeam Echosounder Mills Cross Simulator")
st.markdown(
    "A visual aid to assess the impact of mechanical biases, dynamic IMU motion, and active beam steering on a flat seafloor baseline.")

# --- Session State for pulse characteristics ---
if "pulse_width_ms" not in st.session_state:
    st.session_state.pulse_width_ms = 0.200
if "bandwidth_hz" not in st.session_state:
    st.session_state.bandwidth_hz = 5300.00 # match Tukey
if "cw_lock" not in st.session_state:
    st.session_state.cw_lock = True
if "pulse_tapering" not in st.session_state:
    st.session_state.pulse_tapering = "Tukey"

# --- Set of functions for CW lock toggle ---
def sync_bw_from_tau():
    if st.session_state.cw_lock:
        spectral_ratios = {"Rectangular": 1.0, "Tukey": 1.06, "Hamming": 1.36, "Hann": 1.44}
        ratio = spectral_ratios[st.session_state.pulse_tapering]

        # Use absolute duration for Fourier/Spectral calculations
        abs_tau_s = st.session_state.pulse_width_ms / 1000.0

        # BW = spectral_ratio / absolute_duration
        st.session_state.bandwidth_hz = float(ratio / abs_tau_s)


def sync_tau_from_bw():
    if st.session_state.cw_lock:
        spectral_ratios = {"Rectangular": 1.0, "Tukey": 1.06, "Hamming": 1.36, "Hann": 1.44}
        ratio = spectral_ratios[st.session_state.pulse_tapering]

        # absolute_duration = spectral_ratio / BW
        abs_tau_s = ratio / st.session_state.bandwidth_hz
        st.session_state.pulse_width_ms = float(abs_tau_s * 1000.0)


def sync_from_taper_change():
    # function for taper change alignment
    if st.session_state.cw_lock:
        sync_bw_from_tau()


# --- SIDEBAR INTERFACE ---
st.sidebar.header("Parameters")

# Query beam information
with st.sidebar.container(border=True):
    st.subheader("Interactive Beam Query")
    queried_angle = st.number_input("Query Specific Swath Angle (°)", min_value=-75.0, max_value=75.0, value=45.0, step=1.0)

with st.sidebar.expander("Environment", expanded=False):
    depth = st.number_input("Depth (m)", min_value=1.0, max_value=12000.0, value=25.0, step=1.0)
    c_sound = st.number_input("Sound Speed (m/s)", min_value=1400.0, max_value=1600.0, value=1500.0, step=1.0)
    water_temp = st.number_input("Water Temperature (°C)", min_value=-2.0, max_value=40.0, value=10.0, step=1.0)
    salinity = st.number_input("Salinity (ppt)", min_value=0.0, max_value=50.0, value=35.0, step=1.0)
    ph_level = st.number_input("pH", min_value=5.0, max_value=10.0, value=8.1, step=0.1)

    alpha_placeholder = st.empty()


with st.sidebar.expander("Array Specifications", expanded=False):
    c1, c2 = st.columns(2)
    frequency = st.number_input("Frequency (Hz)", min_value=1000.0, max_value=1000000.0, value=300000.0, step=1000.0)
    tx_beamwidth = c1.number_input("TX BW (Along-Track) (°)", value=0.5, step=0.1)
    rx_beamwidth = c1.number_input("RX BW (Across-Track) (°)", value=1.0, step=0.1)
    tx_across_fan_bw = c2.number_input("TX BW (Across-Track) (°)", value=150.0, step=1.0)
    rx_fore_aft_bw = c2.number_input("RX BW (Along-Track) (°)", value=30.0, step=1.0)

    target_swath_width = st.number_input("Target Swath Coverage (°)", min_value=10.0, max_value=150.0, value=120.0, step=1.0)
    num_sectors = st.selectbox("Number of TX Sectors", options=[1, 2, 3, 4, 5, 8], index=0)

    shading_type = st.selectbox("Array Shading", options=["Uniform", "Hann", "Hamming", "Dolph-Chebyshev"], index=0)
    if shading_type == "Dolph-Chebyshev":
        cheb_attenuation = st.number_input("Desired Sidelobe Suppression (dB)", min_value=1.0, max_value=60.0,
                                           value=30.0, step=1.0)
    else:
        cheb_attenuation = 30.0

# Dynamic Motion
with st.sidebar.expander("Vessel Motion", expanded=False):
    c1, c2, c3 = st.columns(3)
    imu_roll = c1.number_input("Roll (°)", value=0.0, step=1.0)
    imu_pitch = c2.number_input("Pitch (°)", value=0.0, step=1.0)
    imu_yaw = c3.number_input("Yaw (°)", value=0.0, step=1.0)

# Static Mounting Biases
with st.sidebar.expander("Array Mounting Biases", expanded=False):
    st.markdown("**TX Array Biases**")
    c1, c2, c3 = st.columns(3)
    tx_roll_bias = c1.number_input("TX Roll (°)", value=0.0, step=1.0)
    tx_pitch_bias = c2.number_input("TX Pitch (°)", value=0.0, step=1.0)
    tx_yaw_bias = c3.number_input("TX Yaw (°)", value=0.0, step=1.0)

    st.markdown("**RX Array Biases**")
    c4, c5, c6 = st.columns(3)
    rx_roll_bias = c4.number_input("RX Roll (°)", value=0.0, step=1.0)
    rx_pitch_bias = c5.number_input("RX Pitch (°)", value=0.0, step=1.0)
    rx_yaw_bias = c6.number_input("RX Yaw (°)", value=0.0, step=1.0)

# Active Stabilization
with st.sidebar.expander("Active Stabilization & Steering", expanded=False):
    auto_roll = st.checkbox("Active Roll Stabilization (RX)", value=True)
    auto_pitch = st.checkbox("Active Pitch Stabilization (TX)", value=True)
    auto_yaw = st.checkbox("Active Yaw Stabilization (TX)", value=True)
    if not auto_pitch:
        manual_tx_steer = st.number_input("Manual TX Pitch Steer (°)", value=0.0, step=0.1)
    else:
        manual_tx_steer = 0.0

with st.sidebar.expander("Acoustic Energy", expanded=False):
    source_level = st.number_input("Source Level (SL) [dB]", min_value=10.0, max_value=300.0, value=210.0, step=0.1)
    noise_spectrum_level = st.number_input("Ambient Noise Spectrum Level [dB re 1µPa²/Hz]", value=40.0, step=1.0) # maybe move this to environment section
    bs_nadir = st.number_input("Target Strength [dB]", min_value=-60.0, max_value=0.0, value=-20.0, step=1.0)
    apply_tvg = st.checkbox("Apply TVG to Heatmap", value=True)

with st.sidebar.expander("Pulse Specifications", expanded=False):
    st.checkbox("Lock as Continuous Wave (CW)", key="cw_lock", on_change=sync_bw_from_tau,
                help="Locks Bandwidth and Pulse Duration to Fourier inverse relationship.")

    st.number_input("Total Pulse Duration (τ) [ms]", min_value=0.001, max_value=1000.0, step=0.001,
                    key="pulse_width_ms", format="%.3f", on_change=sync_bw_from_tau)

    st.selectbox("Pulse Tapering", ["Rectangular", "Tukey", "Hamming", "Hann"],
                 key="pulse_tapering", on_change=sync_from_taper_change)

    st.number_input("Bandwidth (BW) [Hz]", min_value=10.0, max_value=1000000.0, step=0.01,
                    key="bandwidth_hz", on_change=sync_tau_from_bw)

    # Variables for the Math Engine
    pulse_width_ms = st.session_state.pulse_width_ms
    pulse_tapering = st.session_state.pulse_tapering
    bandwidth_hz = st.session_state.bandwidth_hz

    # --- Fractional Bandwidth Hardware Check ---
    # anything else to check besides frequency?
    fractional_bw_percent = (bandwidth_hz / frequency) * 100.0

    if fractional_bw_percent <= 10.0:
        st.caption(
            f"**Fractional Bandwidth:** {fractional_bw_percent:.1f}% of $f_c$\n\nSupported by standard monolithic transducers")
    elif fractional_bw_percent <= 80.0:
        st.caption(
            f"**Fractional Bandwidth:** :orange[{fractional_bw_percent:.1f}% of $f_c$]\n\nRequires wideband piezo/composite arrays")
    else:
        st.error(
            f"**Fractional Bandwidth:** {fractional_bw_percent:.1f}% of $f_c$\n\nWARNING: Exceeds physical limits of modern transducers (Max 80%).")

    # --- Pulse Alerts ---
    time_bw_product = bandwidth_hz * (pulse_width_ms / 1000.0)

    # Physics check with .99 to account for floating point errors
    if time_bw_product < 0.99:
        st.error(
            f"**Physics Error: TB = {time_bw_product:.2f}**\n\n"
            f"A Time-Bandwidth product below 1.0 violates the Fourier uncertainty principle. "
            f"Transmit of this pulse is impossible."
        )
    # Check if FM
    elif time_bw_product > 1.5 and not st.session_state.cw_lock:
        st.info(
            f"**Active Mode: FM Chirp (TB = {time_bw_product:.1f})**\n\n"
            f"* Range resolution decoupled from pulse duration.\n"
            f"* Receiver matched filter active."
        )
    # Case for valid CW mode
    elif not st.session_state.cw_lock:
        st.success(f"**Active Mode: Custom CW Pulse (TB = {time_bw_product:.2f})**")


with st.sidebar.expander("Acoustic Lobes", expanded=False):
    st.markdown("**Theoretical Beam Patterns (Relative dB)**")
    show_tx_solid = st.checkbox("TX Directivity Pattern (Blue)", value=True)
    if show_tx_solid and num_sectors > 1:
        st.warning(
            "Note: The 3D TX lobe display does not support simultaneous multi-sector visualization. It currently renders the active queried sector only.")
    show_rx_solid = st.checkbox("RX Directivity Pattern (Red)", value=True)
    show_combined_solid = st.checkbox("Combined Effective Beam", value=True)

    st.markdown("**Detection Envelopes (Physical Range)**")
    show_tx_ghost = st.checkbox("TX Ensonified Volume", value=False)
    if show_tx_ghost and num_sectors > 1:
        st.warning(
            "Note: The 3D TX lobe display does not support simultaneous multi-sector visualization. It currently renders the active queried sector only.")
    show_rx_ghost = st.checkbox("RX Noise-Limited Range", value=False)
    show_combined_ghost = st.checkbox("Combined Detection Envelope", value=False)

# --- MATH & GEOMETRY ---
# True Mechanical Orientations (IMU Dynamic Motion + Static Mounting Biases)
true_tx_roll = imu_roll + tx_roll_bias
true_tx_pitch = imu_pitch + tx_pitch_bias
true_tx_yaw = imu_yaw + tx_yaw_bias

true_rx_roll = imu_roll + rx_roll_bias
true_rx_pitch = imu_pitch + rx_pitch_bias
true_rx_yaw = imu_yaw + rx_yaw_bias

# Apply Active Roll Stabilization (Relies only on IMU values, capped at ±10°)
if auto_roll:
    applied_rx_steer = np.clip(imu_roll, -10.0, 10.0)
    array_relative_rx_angle = queried_angle - applied_rx_steer
else:
    array_relative_rx_angle = queried_angle

# Dynamic Sector Steering (Pitch & Yaw Stabilization)
swath_edges = np.linspace(-75.0, 75.0, num_sectors + 1)
sector_limits = [(swath_edges[i], swath_edges[i + 1]) for i in range(num_sectors)]


def get_sector_steering(sector_center_angle):
    """Calculates the unique pitch and yaw steering required for a specific sector"""
    if auto_pitch:
        # Optimal pitch steering changes based on the across-track angle
        pitch_comp_rad = np.arcsin(np.sin(np.radians(-imu_pitch)) * np.cos(np.radians(sector_center_angle)))
        steer_deg = np.degrees(pitch_comp_rad)

        # Clip to mechanical/electronic hardware limits
        steer_deg = np.clip(steer_deg, -10.0, 10.0)
    else:
        steer_deg = manual_tx_steer

    if auto_yaw:
        # Yaw steering math
        yaw_comp_rad = np.arctan(np.tan(np.radians(sector_center_angle)) * np.sin(np.radians(imu_yaw)))
        steer_deg += np.degrees(yaw_comp_rad)

    # Tx sector steering limits
    outer_edge_deg = 0.0
    for s_start, s_end in sector_limits:
        if s_start <= sector_center_angle <= s_end:
            outer_edge_deg = max(abs(s_start), abs(s_end))
            break

    # Mathematical steering limit
    max_allowable_steer = max(0.0, 90.0 - outer_edge_deg - (tx_beamwidth / 2.0))
    steer_deg = np.clip(steer_deg, -max_allowable_steer, max_allowable_steer)

    # This is an attempt to restrict transmit to region within RX listening area. Likely a gross over-simplification of what is actually done.
    max_rx_catch_angle = max(0.0, (rx_fore_aft_bw / 2.0) - (tx_beamwidth / 2.0))
    steer_deg = np.clip(steer_deg, -max_rx_catch_angle, max_rx_catch_angle)

    return np.radians(steer_deg)


# Find which sector the red queried dot belongs to, so it uses the correct physical steering
queried_sector_center = 0.0
for s_start, s_end in sector_limits:
    if s_start <= queried_angle <= s_end:
        queried_sector_center = (s_start + s_end) / 2.0
        break

# Convert variables for the Math Engine
tx_steer_rad = get_sector_steering(queried_sector_center)
tx_steer_angle = np.degrees(tx_steer_rad)  # Preserve for fan geometry
theta_rad = np.radians(array_relative_rx_angle)

# Determine the beamwidth factor based on shading
if shading_type == "Uniform":
    bw_factor = 0.886
elif shading_type == "Hann":
    bw_factor = 1.44
elif shading_type == "Hamming":
    bw_factor = 1.36
elif shading_type == "Dolph-Chebyshev":
    bw_factor = 0.886 * (1.0 + ((cheb_attenuation - 20.0) * 0.016667))

# Calculate physical arrays based on a nominal 1500 m/s sound speed
lambda_nom = 1500.0 / frequency
L_tx = bw_factor * lambda_nom / np.radians(tx_beamwidth)
L_rx = bw_factor * lambda_nom / np.radians(rx_beamwidth)

# Calculate effective beamwidths based on the environmental sound speed slider
wavelength = c_sound / frequency
tx_bw_rad = bw_factor * wavelength / L_tx
rx_bw_rad = bw_factor * wavelength / L_rx

# Apply the Secant Effect for the queried beam
dynamic_tx_bw_rad = tx_bw_rad / np.cos(tx_steer_rad)
dynamic_rx_bw_rad = rx_bw_rad / np.cos(theta_rad)


def calculate_absorption_fg(frequency_hz, T, S, D, pH, c_sound_user):
    """
    Calculates the acoustic absorption coefficient (alpha) in dB/m
    using the full Francois-Garrison (1982) model.
    """
    f = frequency_hz / 1000.0  # Convert Hz to kHz
    T_k = T + 273.15  # Convert Temperature to Kelvin

    # Use the user's manually adjusted sound speed to keep geometry consistent
    c = c_sound_user

    # Boric Acid Contribution (Dominant at low frequencies ~ <10 kHz)
    f1 = 2.8 * np.sqrt(S / 35.0) * (10.0 ** (4.0 - (1245.0 / T_k)))
    A1 = (8.86 / c) * (10.0 ** (0.78 * pH - 5.0))
    alpha_boric = (A1 * f1 * f ** 2) / (f ** 2 + f1 ** 2)

    # Magnesium Sulfate Contribution (Dominant at mid frequencies ~ 10-100 kHz)
    f2 = (8.17 * (10.0 ** (8.0 - (1990.0 / T_k)))) / (1.0 + 0.0018 * (S - 35.0))
    A2 = 21.44 * (S / c) * (1.0 + 0.025 * T)
    P2 = 1.0 - (1.37e-4 * D) + (6.2e-9 * D ** 2)
    alpha_mgso4 = (A2 * P2 * f2 * f ** 2) / (f ** 2 + f2 ** 2)

    # Pure Water Viscosity Contribution (Dominant at high frequencies ~>200 kHz)
    if T <= 20.0:
        A3 = 4.937e-4 - (2.59e-5 * T) + (9.11e-7 * T ** 2) - (1.50e-8 * T ** 3)
    else:
        A3 = 3.964e-4 - (1.146e-5 * T) + (1.45e-7 * T ** 2) - (6.50e-10 * T ** 3)

    P3 = 1.0 - (3.83e-5 * D) + (4.9e-10 * D ** 2)
    alpha_pure = A3 * P3 * f ** 2

    # Total attenuation is the sum of all three components (dB/km)
    alpha_db_km = alpha_boric + alpha_mgso4 + alpha_pure

    # Convert from dB/km to dB/m for local spatial calculations
    return alpha_db_km / 1000.0

display_alpha_db_km = calculate_absorption_fg(frequency, water_temp, salinity, depth, ph_level, c_sound) * 1000.0
alpha_placeholder.info(f"**Absorption Coefficient (α):** {display_alpha_db_km:.2f} dB/km")

# --- 3-Axis Rotation Matrix (Tait-Bryan Yaw-Pitch-Roll) ---
def get_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)

    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    # Standard multiplication order: Z * Y * X
    return np.dot(R_z, np.dot(R_y, R_x))


R_tx_mech = get_rotation_matrix(true_tx_roll, true_tx_pitch, true_tx_yaw)
R_rx_mech = get_rotation_matrix(true_rx_roll, true_rx_pitch, true_rx_yaw)

# Ideal Matrices (Includes IMU motion and assumes no mounting biases)
R_tx_ideal = get_rotation_matrix(imu_roll, imu_pitch, imu_yaw)
R_rx_ideal = get_rotation_matrix(imu_roll, imu_pitch, imu_yaw)


# --- Mills Cross Intersection Solving ---
def solve_mills_cross_intersection(R_tx, R_rx, tx_steer_rad, rx_steer_rad, seafloor_depth):
    """Algebraically solves the 3D intersection using the OE874 Lab D Tp transformation method."""
    # Define ideal vectors and rotate them by their respective mechanical/IMU matrices
    tx_ideal = np.array([1.0, 0.0, 0.0])
    rx_ideal = np.array([0.0, 1.0, 0.0])

    tx_vec = np.dot(R_tx, tx_ideal)
    rx_vec = np.dot(R_rx, rx_ideal)

    # Create new orthonormal basis XYZ' (Tp matrix)
    xp = tx_vec / np.linalg.norm(tx_vec)
    zp = np.cross(tx_vec, rx_vec)
    zp = zp / np.linalg.norm(zp)
    yp = np.cross(zp, xp)
    yp = yp / np.linalg.norm(yp)
    Tp = np.column_stack((xp, yp, zp))

    # Calculate Non-Orthogonality angle (no_a)
    no_a = -np.arcsin(np.clip(np.dot(tx_vec, rx_vec), -1.0, 1.0))

    # Determine components in the local array frame
    y1 = np.sin(rx_steer_rad) / np.cos(no_a)
    y2 = np.sin(tx_steer_rad) * np.tan(no_a)

    rho_hor_sq = (y1 + y2) ** 2 + np.sin(tx_steer_rad) ** 2

    # Check if beams actually intersect (if rho > 1, they do not)
    if rho_hor_sq >= 1.0:
        return np.array([0.0, 0.0, 0.0])

    # Formulate beam vector in XYZ' and transform back to Geo space
    bv_p = np.array([np.sin(tx_steer_rad), y1 + y2, np.sqrt(1.0 - rho_hor_sq)])
    bv_geo = np.dot(Tp, bv_p)

    # Project to flat seafloor
    if bv_geo[2] < 1e-6:
        return np.array([0.0, 0.0, 0.0])

    scale = seafloor_depth / bv_geo[2]
    return bv_geo * scale

def generate_array_weights(N, shading="Hamming", atten_db=30.0):
    """Pre-calculates the amplitude weights for an N-element array."""
    if N <= 1: # single element array catch
        return np.array([1.0])

    n = np.arange(N)
    if shading == "Uniform":
        weights = np.ones(N)
    elif shading == "Hann":
        weights = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    elif shading == "Hamming":
        weights = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    elif shading == "Dolph-Chebyshev":
        weights = chebwin(N, at=atten_db)
    else:
        weights = np.ones(N)

    # Normalize weights so the peak main-lobe amplitude is 1.0
    return weights / np.sum(weights)


@njit(fastmath=True)
def _numba_array_factor(sin_theta, steer_rad, d_lambda, weights):
    """
    An attempt to use optimized discrete array factor calculation using machine code to reduce chugging.
    d_lambda = physical spacing (d) divided by wavelength.
    """
    # Calculate the baseline phase shift based on steering
    phase_shift = 2.0 * np.pi * d_lambda * (sin_theta - np.sin(steer_rad))

    sum_real = 0.0
    sum_imag = 0.0

    # Complex summation across all N elements
    for n in range(len(weights)):
        phase = n * phase_shift
        sum_real += weights[n] * np.cos(phase)
        sum_imag += weights[n] * np.sin(phase)

    # Return the absolute amplitude
    return np.sqrt(sum_real ** 2 + sum_imag ** 2)

# --- Acoustic Directivity and Hardware Math ---
# Physical array elements are locked to the nominal half-wavelength (1500 m/s)
d_spacing_nom = lambda_nom / 2.0
d_lambda_eff = d_spacing_nom / wavelength  # Environmental spacing-to-wavelength ratio

# Theoretical number of elements built into the hardware
true_N_tx = int(np.ceil(L_tx / d_spacing_nom))
true_N_rx = int(np.ceil(L_rx / d_spacing_nom))

# Cap computational elements for Numba to maintain some semblance of UI speed
comp_N_tx = max(1, min(true_N_tx, 300))
comp_N_rx = max(1, min(true_N_rx, 300))

# Pre-calculate distinct weights for TX and RX arrays
tx_weights = generate_array_weights(comp_N_tx, shading=shading_type, atten_db=cheb_attenuation)
rx_weights = generate_array_weights(comp_N_rx, shading=shading_type, atten_db=cheb_attenuation)

def calculate_directivity(v_geo, R_mech, steer_rad, is_tx):
    """Wrapper that dynamically routes TX or RX arrays to the fast Numba math."""
    v_local = np.dot(R_mech.T, v_geo)

    if is_tx:
        sin_theta = v_local[0]
        weights = tx_weights
    else:
        sin_theta = v_local[1]
        weights = rx_weights

    # Call the pre-compiled Numba function using the effective d_lambda
    return _numba_array_factor(sin_theta, steer_rad, d_lambda_eff, weights)


def make_tx_ray(theta_sweep, psi_steer):
    """Intersecting cone math for the TX array"""
    x = np.sin(psi_steer)
    y = np.sin(theta_sweep)
    z_sq = 1.0 - x ** 2 - y ** 2
    z = np.sqrt(max(z_sq, 1e-6))
    return np.array([x, y, z])


def make_rx_ray(theta_steer, phi_acceptance):
    """Intersecting cone math for the RX array"""
    x = np.sin(phi_acceptance)
    y = np.sin(theta_steer)
    z_sq = 1.0 - x ** 2 - y ** 2
    z = np.sqrt(max(z_sq, 1e-6))
    return np.array([x, y, z])


def project_to_flat_bottom(v_ray):
    if v_ray[2] < 1e-6:
        return np.array([v_ray[0] * 1e5, v_ray[1] * 1e5, depth])
    scale = depth / v_ray[2]
    return np.array([v_ray[0] * scale, v_ray[1] * scale, depth])


# Calculate exact 3D nodes
pt_calculated = solve_mills_cross_intersection(R_tx_ideal, R_rx_ideal, tx_steer_rad, theta_rad, depth)
pt_physical = solve_mills_cross_intersection(R_tx_mech, R_rx_mech, tx_steer_rad, theta_rad, depth)

# TX Fan Geometry Construction (Dynamic Multi-Sector Layout)
tx_fwd_psi = tx_steer_rad + (dynamic_tx_bw_rad / 2)
tx_aft_psi = tx_steer_rad - (dynamic_tx_bw_rad / 2)

physical_tx_sectors = []
calculated_tx_sectors = []

for start_angle, end_angle in sector_limits:
    sector_center = (start_angle + end_angle) / 2.0
    sec_steer_rad = get_sector_steering(sector_center)

    # Secant beamwidth expansion for this specific sector
    sec_tx_bw_rad = tx_bw_rad / np.cos(sec_steer_rad)
    sec_fwd_psi = sec_steer_rad + (sec_tx_bw_rad / 2)
    sec_aft_psi = sec_steer_rad - (sec_tx_bw_rad / 2)

    # Physical Fan Sectors
    phys_fwd = [project_to_flat_bottom(np.dot(R_tx_mech, make_tx_ray(t_s, sec_fwd_psi)).flatten()) for t_s in
                np.linspace(np.radians(start_angle), np.radians(end_angle), 25)]
    phys_aft = [project_to_flat_bottom(np.dot(R_tx_mech, make_tx_ray(t_s, sec_aft_psi)).flatten()) for t_s in
                np.linspace(np.radians(start_angle), np.radians(end_angle), 25)]
    physical_tx_sectors.append(phys_fwd + list(reversed(phys_aft)))

    # Ideal Fan Sectors
    calc_fwd = [project_to_flat_bottom(np.dot(R_tx_ideal, make_tx_ray(t_s, sec_fwd_psi)).flatten()) for t_s in
                np.linspace(np.radians(start_angle), np.radians(end_angle), 25)]
    calc_aft = [project_to_flat_bottom(np.dot(R_tx_ideal, make_tx_ray(t_s, sec_aft_psi)).flatten()) for t_s in
                np.linspace(np.radians(start_angle), np.radians(end_angle), 25)]
    calculated_tx_sectors.append(calc_fwd + list(reversed(calc_aft)))

# RX Footprint Geometry Construction
required_acceptance_deg = abs(tx_steer_angle) + (tx_beamwidth / 2.0) + 2.0
rx_acceptance_rad = np.radians(rx_fore_aft_bw / 2.0)

half_rx_bw = dynamic_rx_bw_rad / 2
theta_min = theta_rad - half_rx_bw
theta_max = theta_rad + half_rx_bw

rx_red_perimeter = []
rx_red_perimeter.extend(
    [project_to_flat_bottom(np.dot(R_rx_mech, make_rx_ray(t_s, rx_acceptance_rad)).flatten()) for t_s in
     np.linspace(theta_min, theta_max, 15)])
rx_red_perimeter.extend([project_to_flat_bottom(np.dot(R_rx_mech, make_rx_ray(theta_max, phi)).flatten()) for phi in
                         np.linspace(rx_acceptance_rad, -rx_acceptance_rad, 15)])
rx_red_perimeter.extend(
    [project_to_flat_bottom(np.dot(R_rx_mech, make_rx_ray(t_s, -rx_acceptance_rad)).flatten()) for t_s in
     np.linspace(theta_max, theta_min, 15)])
rx_red_perimeter.extend([project_to_flat_bottom(np.dot(R_rx_mech, make_rx_ray(theta_min, phi)).flatten()) for phi in
                         np.linspace(-rx_acceptance_rad, rx_acceptance_rad, 15)])

rx_full_perimeter = []
rx_full_perimeter.extend(
    [project_to_flat_bottom(np.dot(R_rx_mech, make_rx_ray(t_s, rx_acceptance_rad)).flatten()) for t_s in
     np.linspace(-np.radians(77), np.radians(77), 50)])
rx_full_perimeter.extend(
    [project_to_flat_bottom(np.dot(R_rx_mech, make_rx_ray(np.radians(77), phi)).flatten()) for phi in
     np.linspace(rx_acceptance_rad, -rx_acceptance_rad, 15)])
rx_full_perimeter.extend(
    [project_to_flat_bottom(np.dot(R_rx_mech, make_rx_ray(t_s, -rx_acceptance_rad)).flatten()) for t_s in
     np.linspace(np.radians(77), -np.radians(77), 50)])
rx_full_perimeter.extend(
    [project_to_flat_bottom(np.dot(R_rx_mech, make_rx_ray(-np.radians(77), phi)).flatten()) for phi in
     np.linspace(-rx_acceptance_rad, rx_acceptance_rad, 15)])

rx_full_x = [p[0] for p in rx_full_perimeter]
rx_full_y = [p[1] for p in rx_full_perimeter]
rx_full_z = [p[2] for p in rx_full_perimeter]

# --- Calculate Sounding Patch ---
tx_edge_fwd = solve_mills_cross_intersection(R_tx_mech, R_rx_mech, tx_fwd_psi, theta_rad, depth)
tx_edge_aft = solve_mills_cross_intersection(R_tx_mech, R_rx_mech, tx_aft_psi, theta_rad, depth)

rx_edge_max = solve_mills_cross_intersection(R_tx_mech, R_rx_mech, tx_steer_rad, theta_max, depth)
rx_edge_min = solve_mills_cross_intersection(R_tx_mech, R_rx_mech, tx_steer_rad, theta_min, depth)

has_overlap = all(np.linalg.norm(p) > 1e-3 for p in [tx_edge_fwd, tx_edge_aft, rx_edge_max, rx_edge_min])

patch_points = []
if has_overlap:
    vec_tx = (tx_edge_fwd - tx_edge_aft) / 2.0
    vec_rx = (rx_edge_max - rx_edge_min) / 2.0

    angles = np.linspace(0, 2 * np.pi, 64)
    for alpha in angles:
        pt = pt_physical + vec_tx * np.cos(alpha) + vec_rx * np.sin(alpha)
        patch_points.append(pt)

    a = np.linalg.norm(vec_tx)
    b = np.linalg.norm(vec_rx)
    patch_area = np.pi * a * b

    # --- Pulse Resolution & Tapering Logic ---

    # Equivalent Rectangular Duration Ratios for Spatial Resolution (Energy)
    erd_ratios = {"Rectangular": 1.0, "Tukey": 0.938, "Hamming": 0.397, "Hann": 0.375}
    effective_pulse_duration_s = (pulse_width_ms / 1000.0) * erd_ratios.get(pulse_tapering, 1.0)

    # Spectral Broadening Ratios for Bandwidth (Fourier)
    spectral_ratios = {"Rectangular": 1.0, "Tukey": 1.06, "Hamming": 1.36, "Hann": 1.44}

    # Calculate TB Product using absolute duration
    time_bw_product = bandwidth_hz * (pulse_width_ms / 1000.0)

    # Range Resolution Logic (FM vs CW)
    if time_bw_product > 1.5:
        range_res_m = c_sound / (2.0 * bandwidth_hz)  # FM Chirp Compressed
    else:
        range_res_m = c_sound * effective_pulse_duration_s / 2.0  # CW Tapered Spatial Resolution

    # --- Receiver Bandwidth Noise Escalation ---
    # Convert baseline spectrum noise into total receiver band noise (10 * log10(BW))
    total_noise_level = noise_spectrum_level + 10 * np.log10(bandwidth_hz)

    # Project pulse thickness onto flat seafloor
    incidence_angle = abs(theta_rad)
    projected_pulse_width = range_res_m / max(1e-6, np.sin(incidence_angle))

    # Geometric across-track length of the patch
    geo_width = 2.0 * b

    is_pulse_limited = False
    active_patch_area = patch_area

    # Check if pulse is shorter than footprint
    if projected_pulse_width < geo_width and incidence_angle >= np.radians(2.0):
        is_pulse_limited = True
        active_patch_area = np.pi * a * (projected_pulse_width / 2.0)

    # Calculate number of pulse slices
    num_cells = max(1, int(geo_width / projected_pulse_width))
    num_cells = min(num_cells, 60)  # Cap for rendering performance

    # Build 3D mesh polygons for pulse bands
    pulse_slice_meshes = []
    y_edges = np.linspace(-1.0, 1.0, num_cells + 1)
    colors = ['rgba(0, 255, 200, 0.65)', 'rgba(150, 0, 255, 0.65)']  # Alternating Teal and Purple

    for k in range(num_cells):
        v_start = y_edges[k]
        v_end = y_edges[k + 1]

        # Ensure outer arcs have enough vertices to look smooth
        num_pts = max(5, int(40 / num_cells))
        v_sweep = np.linspace(v_start, v_end, num_pts)

        # Calculate elliptical boundaries
        u_right = np.sqrt(np.clip(1.0 - v_sweep ** 2, 0, 1))
        u_left = -np.sqrt(np.clip(1.0 - v_sweep[::-1] ** 2, 0, 1))

        v_poly = np.concatenate([v_sweep, v_sweep[::-1]])
        u_poly = np.concatenate([u_right, u_left])

        poly_x, poly_y, poly_z = [], [], []
        for u, v in zip(u_poly, v_poly):
            pt = pt_physical + vec_tx * u + vec_rx * v
            poly_x.append(pt[0])
            poly_y.append(pt[1])
            poly_z.append(pt[2])

        pulse_slice_meshes.append(go.Mesh3d(
            x=poly_x, y=poly_y, z=poly_z,
            color=colors[k % 2],
            delaunayaxis='z',
            hoverinfo='skip',
            name='Pulse Duration Bands' if k == 0 else None,
            showlegend=(k == 0)
        ))

else:
    patch_area = 0.0
    active_patch_area = 0.0
    range_res_m = 0.0
    is_pulse_limited = False
    num_cells = 0
    pulse_slice_meshes = []

# --- METRICS & VALUES ---
delta_x = pt_physical[0] - pt_calculated[0]
delta_y = pt_physical[1] - pt_calculated[1]

tx_edge_fwd = solve_mills_cross_intersection(R_tx_mech, R_rx_mech, tx_fwd_psi, theta_rad, depth)
tx_edge_aft = solve_mills_cross_intersection(R_tx_mech, R_rx_mech, tx_aft_psi, theta_rad, depth)

if np.linalg.norm(tx_edge_fwd) > 0 and np.linalg.norm(tx_edge_aft) > 0:
    tx_x_width = np.linalg.norm(tx_edge_fwd - tx_edge_aft)
    tx_status = "Yes" if has_overlap else "No"
else:
    tx_x_width = 0.0
    tx_status = "No"

# Hardware boundary checks
if np.linalg.norm(pt_physical) > 0:
    v_pt_phys = pt_physical / np.linalg.norm(pt_physical)

    # RX Check
    v_local_rx = np.dot(R_rx_mech.T, v_pt_phys)
    actual_rx_fore_aft_angle = np.degrees(np.arcsin(np.clip(v_local_rx[0], -1.0, 1.0)))
    rx_status = "Yes" if abs(actual_rx_fore_aft_angle) <= (rx_fore_aft_bw / 2.0) else ":red[No]"

    # TX Check
    v_local_tx = np.dot(R_tx_mech.T, v_pt_phys)
    actual_tx_across_angle = np.degrees(np.arcsin(np.clip(v_local_tx[1], -1.0, 1.0)))
    tx_status = "Yes" if abs(actual_tx_across_angle) <= (tx_across_fan_bw / 2.0) else ":red[No]"
else:
    rx_status = "Invalid"
    tx_status = "Invalid"

st.subheader(f"Intersection Metrics for Queried Beam ({queried_angle}°)")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Along Dev (X)", f"{delta_x:.2f} m")
col2.metric("Across Dev (Y)", f"{delta_y:.2f} m")
col3.metric("Inside TX Fan?", tx_status)
col4.metric("Inside RX Listening Area?", rx_status)
col5.metric("Along Track Patch Width", f"{tx_x_width:.2f} m")
if is_pulse_limited:
    col6.metric("Active Area (Pulse-Limited)", f"{active_patch_area:.2f} m²")
else:
    col6.metric("Active Area (Beam-Limited)", f"{active_patch_area:.2f} m²")

# --- Calculate Transmission Loss and Intensity for Queried Beam---
# Convert user's sidebar items to standard SI base units
tau_sec = pulse_width_ms / 1000.0
bw_hz = bandwidth_hz

# --- Calculate Transmission Loss and Intensity for Queried Beam---
# Calculate Signal Processing Gain (PG)
if time_bw_product > 1.5:
    processing_gain = 10 * np.log10(time_bw_product)
else:
    processing_gain = 0.0

# Ensure total_noise_level is globally defined for downstream math
total_noise_level = noise_spectrum_level + 10 * np.log10(bandwidth_hz)

slant_range_m = 0.0
alpha_db_m = 0.0
spreading_loss = 0.0
absorption_loss = 0.0
two_way_tl = 0.0
dynamic_ts = 0.0
relative_intensity = 0.0
absolute_pressure = 0.0
status_text = ":red[Not Detected]"
tvg_display = "N/A"

if np.linalg.norm(pt_physical) > 0 and patch_area > 0:
    slant_range_m = np.linalg.norm(pt_physical)
    alpha_db_m = calculate_absorption_fg(frequency, water_temp, salinity, depth, ph_level, c_sound)

    spreading_loss = 40 * np.log10(slant_range_m)
    absorption_loss = 2 * alpha_db_m * slant_range_m
    two_way_tl = spreading_loss + absorption_loss

    lambert_angular_decay = 10 * np.log10(np.cos(np.radians(queried_angle)) ** 2 + 1e-12)
    area_scattering = 10 * np.log10(active_patch_area + 1e-12)
    dynamic_ts = bs_nadir + lambert_angular_decay + area_scattering

    relative_intensity = dynamic_ts - two_way_tl
    absolute_pressure = source_level + relative_intensity + processing_gain

    if absolute_pressure > total_noise_level:
        status_text = ":green[Detected]"
        tvg_display = f"{dynamic_ts:.1f} dB"
    else:
        status_text = ":red[Not Detected]"
        tvg_display = "N/A (No Signal)"

acol1, acol2, acol3, acol4, acol5, acol6 = st.columns(6)
acol1.metric("Detection Status (Range Only)", status_text)
acol2.metric("TVG Corrected Return Intensity", tvg_display)
acol3.metric("Raw Return (Absolute Pressure)", f"{absolute_pressure:.1f} dB")


st.markdown("---")
st.subheader("**Theoretical Array Specifications (Nominal 1500 m/s)**")

# Calculate UI variables matching the math engine
lambda_nom_ui = 1500.0 / frequency
if shading_type == "Uniform":
    bw_factor_ui = 0.886
elif shading_type == "Hann":
    bw_factor_ui = 1.44
elif shading_type == "Hamming":
    bw_factor_ui = 1.36
elif shading_type == "Dolph-Chebyshev":
    bw_factor_ui = 0.886 * (1.0 + ((cheb_attenuation - 20.0) * 0.016667))

L_tx_ui = bw_factor_ui * lambda_nom_ui / np.radians(tx_beamwidth)
L_rx_ui = bw_factor_ui * lambda_nom_ui / np.radians(rx_beamwidth)

N_tx_ui = int(np.ceil(L_tx_ui / (lambda_nom_ui / 2.0)))
N_rx_ui = int(np.ceil(L_rx_ui / (lambda_nom_ui / 2.0)))

# Calculate Effective Beamwidth for UI
wav_env_ui = c_sound / frequency
eff_tx_deg = np.degrees(bw_factor_ui * wav_env_ui / L_tx_ui)
eff_rx_deg = np.degrees(bw_factor_ui * wav_env_ui / L_rx_ui)

# Calculate deltas and hide them if the rounded difference is 0.00
tx_diff = eff_tx_deg - tx_beamwidth
tx_delta = f"{tx_diff:.2f}°" if abs(tx_diff) >= 0.005 else None

rx_diff = eff_rx_deg - rx_beamwidth
rx_delta = f"{rx_diff:.2f}°" if abs(rx_diff) >= 0.005 else None

# 6 column layout for array specs metrics
hw1, hw2, hw3, hw4, hw5, hw6 = st.columns(6)
hw1.metric("TX Length", f"{L_tx_ui:.2f} m")
hw2.metric("RX Length", f"{L_rx_ui:.2f} m")
hw3.metric("TX Elements", f"{N_tx_ui}")
hw4.metric("RX Elements", f"{N_rx_ui}")
hw5.metric("Effective TX BW", f"{eff_tx_deg:.2f}°", delta=tx_delta, delta_color="inverse")
hw6.metric("Effective RX BW", f"{eff_rx_deg:.2f}°", delta=rx_delta, delta_color="inverse")

# --- GRAPH TOGGLES ---
st.markdown("---")
st.subheader("Visualization Overlays")

# condense toggles to left side of screen
t_col1, t_col2, t_col3, t_col4, t_col5, t_col6, t_col7, t_col8, spacer = st.columns([1.5, 1.5, 1.2, 1.2, 2.0, 2.0, 2.0, 2.0, 3.0])

show_ideal_tx = t_col1.checkbox("Ideal TX Sectors", value=True)
show_actual_tx = t_col2.checkbox("Actual TX Sectors", value=True)
show_rx_bowtie = t_col3.checkbox("RX Bowtie", value=True)
show_rx_red = t_col4.checkbox("RX Footprint", value=True)
show_ideal_soundings = t_col5.checkbox("Ideal Soundings (100 beams)", value=False)
show_actual_soundings = t_col6.checkbox("Actual Soundings (100 beams)", value=False)
show_heatmap = t_col7.checkbox("Show Seafloor Heatmap (dB)", value=True)


# --- Equidistant beam math (100 beams for simplicity) ---
# This is currently NOT properly functional for all axes of motion stabilization.
ideal_sounding_dots = []
actual_sounding_dots = []

if show_ideal_soundings or show_actual_soundings:
    per_side_target = target_swath_width / 2.0
    max_y = depth * np.tan(np.radians(per_side_target))

    # Create 100 target coordinates
    beam_y_coords = np.linspace(-max_y, max_y, 100)

    for y_coord in beam_y_coords:
        # Back-calculate the nominal angle to determine which sector panel this target coordinate falls into
        b_angle_nominal = np.degrees(np.arctan(y_coord / depth))

        sector_center = 0.0
        for s_start, s_end in sector_limits:
            if s_start <= b_angle_nominal <= s_end:
                sector_center = (s_start + s_end) / 2.0
                break

        # Get the steering angle command for this specific sector panel
        steer_rad = get_sector_steering(sector_center)

        # Back projection and swath truncation
        # Define 3D target coordinate on the flat seafloor
        v_target_global = np.array([depth * np.tan(steer_rad), y_coord, depth])

        # Push global target backwards into the array's local frame
        if auto_roll:
            v_target_local = np.dot(R_rx_ideal.T, v_target_global)
        else:
            v_target_local = v_target_global

        # Calculate electronic receive angle needed to hit local coordinate
        b_rad = np.arcsin(np.clip(v_target_local[1] / np.linalg.norm(v_target_local), -1.0, 1.0))

        max_rx_hardware_angle = 75.0

        # Drop beam to dynamically truncate the swath if exceeded
        if abs(np.degrees(b_rad)) > max_rx_hardware_angle:
            continue

        # Ideal Soundings
        if show_ideal_soundings:
            pt_id = solve_mills_cross_intersection(R_tx_ideal, R_rx_ideal, steer_rad, b_rad, depth)
            if np.linalg.norm(pt_id) > 0:
                ideal_sounding_dots.append(pt_id)

        # Actual Soundings
        if show_actual_soundings:
            pt_ac = solve_mills_cross_intersection(R_tx_mech, R_rx_mech, steer_rad, b_rad, depth)
            if np.linalg.norm(pt_ac) > 0:
                actual_sounding_dots.append(pt_ac)

# --- 3D Visualization ---
fig = go.Figure()

# --- Seafloor Heatmap ---
if show_heatmap and np.linalg.norm(pt_physical) > 0:
    span_factor = 8.0  # Number of beamwidths to display

    nominal_range_x = depth * np.tan(np.radians(tx_beamwidth * span_factor))
    nominal_range_y = depth * np.tan(np.radians(rx_beamwidth * span_factor)) / (np.cos(theta_rad) ** 2)

    # Identify larger small axis beamwidth and set for square map rendering
    grid_range = max(10.0, nominal_range_x, nominal_range_y)

    grid_range_x = grid_range
    grid_range_y = grid_range

    # Generate centered on actual sounding point
    x_g = np.linspace(pt_physical[0] - grid_range_x, pt_physical[0] + grid_range_x, 100)
    y_g = np.linspace(pt_physical[1] - grid_range_y, pt_physical[1] + grid_range_y, 100)
    X_grid, Y_grid = np.meshgrid(x_g, y_g)
    Z_grid = np.full_like(X_grid, depth)
    Intensity_dB = np.zeros_like(X_grid)

    # Calculate alpha once for the whole grid
    alpha_db_m = calculate_absorption_fg(frequency, water_temp, salinity, depth, ph_level, c_sound)

    # Dynamically anchor the color scale to the nadir values so it never blanks out
    nadir_tl = 40 * np.log10(depth) + 2 * alpha_db_m * depth

    if apply_tvg:
        cmax_val = bs_nadir
        cmin_val = bs_nadir - 50.0
        cbar_title = "TVG Corrected Intensity (dB)"
    else:
        # Raw Intensity strips away SL, leaving just the environmental penalties
        cmax_val = bs_nadir - nadir_tl
        cmin_val = cmax_val - 60.0
        cbar_title = "Raw Intensity (dB)"

    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            P = np.array([X_grid[i, j], Y_grid[i, j], Z_grid[i, j]])
            R = np.linalg.norm(P)

            if R == 0:
                continue

            v_geo = P / R

            # Array Directivity
            D_tx = calculate_directivity(v_geo, R_tx_mech, tx_steer_rad, is_tx=True)
            D_rx = calculate_directivity(v_geo, R_rx_mech, theta_rad, is_tx=False)
            I_linear = np.abs(D_tx * D_rx)
            DI_dB = 20 * np.log10(I_linear + 1e-12)

            # Transmission Loss
            two_way_tl = 40 * np.log10(R) + 2 * alpha_db_m * R

            # Lambertian Target Strength
            cos_theta = abs(Z_grid[i, j]) / R
            lambert_decay = 10 * np.log10(cos_theta ** 2 + 1e-12)

            heatmap_area_scattering = 10 * np.log10(patch_area + 1e-12)
            pixel_ts = bs_nadir + lambert_decay + heatmap_area_scattering

            # Calculate absolute physical pressure for this specific pixel
            pixel_absolute_pressure = source_level - two_way_tl + pixel_ts + DI_dB + processing_gain
            raw_intensity = pixel_ts - two_way_tl + DI_dB

            if pixel_absolute_pressure <= total_noise_level:
                Intensity_dB[i, j] = cmin_val
            else:
                if apply_tvg:
                    Intensity_dB[i, j] = pixel_ts + DI_dB
                else:
                    Intensity_dB[i, j] = raw_intensity

    # Plot the Surface
    fig.add_trace(go.Surface(
        x=X_grid, y=Y_grid, z=Z_grid,
        surfacecolor=Intensity_dB,
        colorscale='Jet',
        cmin=cmin_val, cmax=cmax_val,
        name='Seafloor Footprint',
        showscale=True,
        colorbar=dict(title=cbar_title, x=0.85, len=0.5)
    ))

# --- 3D Acoustic Lobes ---
if (show_tx_solid or show_tx_ghost or show_rx_solid or show_rx_ghost or show_combined_solid or show_combined_ghost) and np.linalg.norm(pt_physical) > 0:
    # Find max Two-Way Transmission Loss
    max_allowable_tl = source_level + dynamic_ts + processing_gain - total_noise_level

    # Calculate absorption for lobes
    lobe_alpha_db_m = calculate_absorption_fg(frequency, water_temp, salinity, depth, ph_level, c_sound)

    # Iterative Binary Solver to find Maximum Detection Range (R_max)
    if max_allowable_tl <= 0:
        lobe_scale = 1.0  # Signal is DOA
    else:
        r_min = 1.0
        r_max = 20000.0

        for _ in range(50):
            r_mid = (r_min + r_max) / 2.0
            tl_test = 40 * np.log10(r_mid) + 2 * lobe_alpha_db_m * r_mid

            if tl_test < max_allowable_tl:
                r_min = r_mid
            else:
                r_max = r_mid

        lobe_scale = r_mid


    def generate_native_lobe(is_tx, color_scale, name, mode="Ghost"):
        """Generates a mathematically pure 3D polar directivity pattern."""

        # Dynamically center the grid on the steered beam
        if is_tx:
            v_center = np.pi / 2 - tx_steer_rad
        else:
            v_center = np.pi / 2 - theta_rad

        # Using odd numbers (151, 201) to ensure exact center point is sampled
        u = np.linspace(0, np.pi, 151)
        v = np.linspace(v_center - np.radians(45), v_center + np.radians(45), 201)
        U, V = np.meshgrid(u, v)

        if is_tx:
            X_unit = np.cos(V)
            Y_unit = np.sin(V) * np.cos(U)
            Z_unit = np.sin(V) * np.sin(U)
        else:
            X_unit = np.sin(V) * np.cos(U)
            Y_unit = np.cos(V)
            Z_unit = np.sin(V) * np.sin(U)

        R_linear = np.zeros_like(X_unit)

        for i in range(X_unit.shape[0]):
            for j in range(X_unit.shape[1]):
                v_geo = np.array([X_unit[i, j], Y_unit[i, j], Z_unit[i, j]])
                if is_tx:
                    D = calculate_directivity(v_geo, R_tx_mech, tx_steer_rad, is_tx=True)
                else:
                    D = calculate_directivity(v_geo, R_rx_mech, theta_rad, is_tx=False)
                R_linear[i, j] = np.abs(D)

        # Normalize and map to dB scale (-40dB Floor) for color and Solid mode
        R_dB = np.clip(20 * np.log10(R_linear + 1e-12), -40, 0)

        # --- Choose radial scale based on physical mode ---
        if mode == "Ghost":
            # Uses linear acoustic pressure scaling. Side lobes accurately reflect their limited physical penetration in the water column.
            R_final = R_linear * lobe_scale
            surface_opacity = 0.15

        elif mode == "Solid":
            # Uses linear-dB mapping so side lobes remain visually distinct. Theoretical pattern.
            R_physical_radius = (R_dB + 40.0) / 40.0
            R_final = R_physical_radius * depth
            surface_opacity = 0.8

        # Apply final calculated scale point-by-point
        X_lobe = X_unit * R_final
        Y_lobe = Y_unit * R_final
        Z_lobe = Z_unit * R_final

        return go.Surface(
            x=X_lobe, y=Y_lobe, z=Z_lobe,
            surfacecolor=R_dB,  # Color map always remains in dB for contrast
            colorscale=color_scale,
            cmin=-40, cmax=0,
            name=f"{name} ({mode})",
            showscale=False,
            opacity=surface_opacity
        )

    # Add TX Traces
    if show_tx_solid:
        fig.add_trace(generate_native_lobe(is_tx=True, color_scale='Blues', name='TX Lobe', mode="Solid"))
    if show_tx_ghost:
        fig.add_trace(generate_native_lobe(is_tx=True, color_scale='Blues', name='TX Lobe', mode="Ghost"))

    # Add RX Traces
    if show_rx_solid:
        fig.add_trace(generate_native_lobe(is_tx=False, color_scale='Reds', name='RX Lobe', mode="Solid"))
    if show_rx_ghost:
        fig.add_trace(generate_native_lobe(is_tx=False, color_scale='Reds', name='RX Lobe', mode="Ghost"))


    def generate_combined_lobe(mode="Ghost"):
        """Generates the mathematically pure combined beam."""

        beam_vec = pt_physical / np.linalg.norm(pt_physical)
        azimuth_center = np.arctan2(beam_vec[1], beam_vec[0])
        elevation_center = np.arccos(beam_vec[2])

        span = np.radians(8.0)
        u_comb = np.linspace(azimuth_center - span, azimuth_center + span, 81)
        v_comb = np.linspace(max(0, elevation_center - span), min(np.pi / 2, elevation_center + span), 81)
        U_comb, V_comb = np.meshgrid(u_comb, v_comb)

        X_comb = np.sin(V_comb) * np.cos(U_comb)
        Y_comb = np.sin(V_comb) * np.sin(U_comb)
        Z_comb = np.cos(V_comb)
        R_comb_linear = np.zeros_like(X_comb)

        for i in range(X_comb.shape[0]):
            for j in range(X_comb.shape[1]):
                v_geo = np.array([X_comb[i, j], Y_comb[i, j], Z_comb[i, j]])
                D_tx = calculate_directivity(v_geo, R_tx_mech, tx_steer_rad, is_tx=True)
                D_rx = calculate_directivity(v_geo, R_rx_mech, theta_rad, is_tx=False)
                R_comb_linear[i, j] = np.abs(D_tx * D_rx)

        # Normalize to full dB scale
        R_dB = np.clip(20 * np.log10(R_comb_linear + 1e-12), -40, 0)

        # --- Choose radial scale based on mode ---
        if mode == "Ghost":
            # Two-way acoustic propagation boundary scale
            R_final = R_comb_linear * lobe_scale
            surface_opacity = 0.25

        elif mode == "Solid":
            # Polar plot scale
            R_radius = (R_dB + 40.0) / 40.0
            R_final = R_radius * depth
            surface_opacity = 0.85

        X_final = X_comb * R_final
        Y_final = Y_comb * R_final
        Z_final = Z_comb * R_final

        return go.Surface(
            x=X_final, y=Y_final, z=Z_final,
            surfacecolor=R_dB, colorscale='Viridis', cmin=-40, cmax=0,
            name=f'Combined Lobe ({mode})', showscale=False, opacity=surface_opacity
        )

    if show_combined_solid:
        fig.add_trace(generate_combined_lobe(mode="Solid"))
    if show_combined_ghost:
        fig.add_trace(generate_combined_lobe(mode="Ghost"))

# Add Actual TX Footprint
if show_actual_tx:
    actual_colors = ['darkorange', 'gold']
    for idx, sector_pts in enumerate(physical_tx_sectors):
        color = actual_colors[idx % len(actual_colors)]
        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in sector_pts] + [sector_pts[0][0]],
            y=[p[1] for p in sector_pts] + [sector_pts[0][1]],
            z=[p[2] for p in sector_pts] + [sector_pts[0][2]],
            mode='lines', line=dict(color=color, width=4),
            name='Actual TX Sectors' if idx == 0 else None,
            showlegend=(idx == 0)
        ))

# Add Ideal TX Footprint
if show_ideal_tx:
    ideal_colors = ['darkblue', 'blue']
    for idx, sector_pts in enumerate(calculated_tx_sectors):
        color = ideal_colors[idx % len(ideal_colors)]
        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in sector_pts] + [sector_pts[0][0]],
            y=[p[1] for p in sector_pts] + [sector_pts[0][1]],
            z=[p[2] for p in sector_pts] + [sector_pts[0][2]],
            mode='lines', line=dict(color=color, width=3, dash='dash'),
            name='Ideal TX Sectors' if idx == 0 else None,
            showlegend=(idx == 0)
        ))

# --- RX Listening Region ---
if show_rx_bowtie:
    theta_sweep = np.linspace(-np.radians(80.0), np.radians(80.0), 60)
    phi_sweep = np.linspace(-rx_acceptance_rad, rx_acceptance_rad, 25)

    THETA, PHI = np.meshgrid(theta_sweep, phi_sweep)

    X_carpet = np.zeros_like(THETA)
    Y_carpet = np.zeros_like(THETA)
    Z_carpet = np.zeros_like(THETA)
    C_carpet = np.zeros_like(THETA)

    for i in range(THETA.shape[0]):
        for j in range(THETA.shape[1]):
            t_val = THETA[i, j]
            p_val = PHI[i, j]

            # Generate the ray and project it to the seafloor
            ray_local = make_rx_ray(t_val, p_val)
            pt = project_to_flat_bottom(np.dot(R_rx_mech, ray_local).flatten())

            X_carpet[i, j] = pt[0]
            Y_carpet[i, j] = pt[1]
            Z_carpet[i, j] = pt[2]

            # Calculate 2D Acoustic Taper
            # Along-track fade: Cosine taper down to the acceptance limits
            fade_along = np.cos((np.pi / 2.0) * (p_val / rx_acceptance_rad))

            # Across-track fade: Sensitivity drops naturally as projection area shrinks (~cos of angle)
            fade_across = np.cos(t_val)

            # Combine them for a smooth decay envelope
            C_carpet[i, j] = max(0.0, fade_along * fade_across)

    fade_green_scale = [
        [0.0, 'rgba(34, 139, 34, 0.0)'],  # 0% opacity at the edges
        [0.5, 'rgba(34, 139, 34, 0.2)'],  # 20% opacity mid-way
        [1.0, 'rgba(34, 139, 34, 0.5)']  # 50% opacity at the center axis
    ]

    fig.add_trace(go.Surface(
        x=X_carpet, y=Y_carpet, z=Z_carpet,
        surfacecolor=C_carpet,
        colorscale=fade_green_scale,
        cmin=0, cmax=1,
        name='RX Listening Area',
        showscale=False,
        hoverinfo='skip'
    ))

# Add RX Footprint (Red Strip)
if show_rx_red:
    fig.add_trace(go.Scatter3d(
        x=[p[0] for p in rx_red_perimeter] + [rx_red_perimeter[0][0]],
        y=[p[1] for p in rx_red_perimeter] + [rx_red_perimeter[0][1]],
        z=[p[2] for p in rx_red_perimeter] + [rx_red_perimeter[0][2]],
        mode='lines', line=dict(color='red', width=4), name='RX Footprint (Red Strip)'
    ))

if has_overlap:
    fig.add_trace(go.Scatter3d(
        x=[p[0] for p in patch_points] + [patch_points[0][0]],
        y=[p[1] for p in patch_points] + [patch_points[0][1]],
        z=[p[2] for p in patch_points] + [patch_points[0][2]],
        mode='lines',
        line=dict(color='purple', width=5),
        name='Sounding Patch'
    ))

# --- Visual Representation of Arrays and Bow Vector ---
# Arbitrary value to maintain visualization at any scale
array_visual_length = depth * 0.15

# Physical TX Array (Aligned along X-axis locally, rotated by R_tx_mech)
tx_local_start = np.array([-array_visual_length / 2.0, 0.0, 0.0])
tx_local_end = np.array([array_visual_length / 2.0, 0.0, 0.0])
tx_global_start = np.dot(R_tx_mech, tx_local_start)
tx_global_end = np.dot(R_tx_mech, tx_local_end)

fig.add_trace(go.Scatter3d(
    x=[tx_global_start[0], tx_global_end[0]],
    y=[tx_global_start[1], tx_global_end[1]],
    z=[tx_global_start[2], tx_global_end[2]],
    mode='lines',
    line=dict(color='blue', width=10),
    name='Physical TX Array (Keel)'
))

# Physical RX Array (Aligned along Y-axis locally, rotated by R_rx_mech)
rx_local_start = np.array([0.0, -array_visual_length / 2.0, 0.0])
rx_local_end = np.array([0.0, array_visual_length / 2.0, 0.0])
rx_global_start = np.dot(R_rx_mech, rx_local_start)
rx_global_end = np.dot(R_rx_mech, rx_local_end)

fig.add_trace(go.Scatter3d(
    x=[rx_global_start[0], rx_global_end[0]],
    y=[rx_global_start[1], rx_global_end[1]],
    z=[rx_global_start[2], rx_global_end[2]],
    mode='lines',
    line=dict(color='red', width=10),
    name='Physical RX Array (Beam)'
))

# Vessel Bow Arrow (Follows where bow would point under motion)
fwd_visual_length = depth * 0.18
fwd_local = np.array([fwd_visual_length, 0.0, 0.0])
fwd_global = np.dot(R_tx_ideal, fwd_local)

# Arrow Shaft
fig.add_trace(go.Scatter3d(
    x=[0, fwd_global[0]],
    y=[0, fwd_global[1]],
    z=[0, fwd_global[2]],
    mode='lines',
    line=dict(color='black', width=5),
    name='Vessel Forward Direction'
))

# 3D Arrowhead Cone
fig.add_trace(go.Cone(
    x=[fwd_global[0]], y=[fwd_global[1]], z=[fwd_global[2]],
    u=[fwd_global[0]], v=[fwd_global[1]], w=[fwd_global[2]],
    sizemode="absolute",
    sizeref=depth * 0.03,
    colorscale=[[0, 'black'], [1, 'black']],
    showscale=False,
    name='Forward Arrowhead',
    hoverinfo='skip'
))

fig.add_trace(go.Scatter3d(x=[0, pt_calculated[0]], y=[0, pt_calculated[1]], z=[0, pt_calculated[2]], mode='lines',
                           line=dict(color='blue', width=4), name='Ideal Pointing Vector'))
fig.add_trace(go.Scatter3d(x=[0, pt_physical[0]], y=[0, pt_physical[1]], z=[0, pt_physical[2]], mode='lines',
                           line=dict(color='red', width=6), name='Actual Pointing Vector'))

fig.add_trace(go.Scatter3d(x=[pt_calculated[0]], y=[pt_calculated[1]], z=[pt_calculated[2]], mode='markers',
                           marker=dict(color='blue', size=6), name='Ideal Sounding'))

# --- Dynamic Pulse Band Surface (Resolution Cells) ---
if has_overlap and len(pulse_slice_meshes) > 0:
    for mesh in pulse_slice_meshes:
        fig.add_trace(mesh)


# --- Visual Angle Reference Grid Tracking TX Sectors ---
ref_angles = np.linspace(-75, 75, 11, dtype=int)  # Label every 15 degrees
lbl_x, lbl_y, lbl_z, lbl_text = [], [], [], []

for i, ang in enumerate(ref_angles):
    # Dynamically find which transmit sector panel this reference angle belongs to
    sector_center = 0.0
    for s_start, s_end in sector_limits:
        if s_start <= ang <= s_end:
            sector_center = (s_start + s_end) / 2.0
            break

    # Get actual active stabilization steering for this specific sector
    sec_steer_rad = get_sector_steering(sector_center)

    # Build the local ray and rotate it via the actual TX mechanical matrix
    v_ray = make_tx_ray(np.radians(ang), sec_steer_rad)
    v_rot = np.dot(R_tx_mech, v_ray)

    # Project down to intersect the flat seafloor
    if v_rot[2] > 1e-6:
        scale = depth / v_rot[2]
        x_rot = v_rot[0] * scale
        y_rot = v_rot[1] * scale
    else:
        x_rot, y_rot = 0, 0

    fig.add_trace(go.Scatter3d(
        x=[0, x_rot], y=[0, y_rot], z=[0, depth],
        mode='lines',
        line=dict(color='gray', width=2),
        opacity=0.3,
        name='Angle Reference Grid' if i == 0 else None,
        showlegend=False,
        legendgroup='grid',
        hoverinfo='skip'
    ))

    # Save label coordinates
    lbl_x.append(x_rot)
    lbl_y.append(y_rot)
    lbl_z.append(depth)
    lbl_text.append(f"{ang}°")

    # Add the angle text labels to the seafloor
    fig.add_trace(go.Scatter3d(
        x=lbl_x, y=lbl_y, z=lbl_z,
        mode='text', text=lbl_text, textfont=dict(color='gray', size=12),
        name='Angle Labels',
        showlegend=False,
        hoverinfo='skip'
    ))

# Add 100 Ideal Sounding Points (blue dots matching ideal sectors)
if show_ideal_soundings and len(ideal_sounding_dots) > 0:
    fig.add_trace(go.Scatter3d(
        x=[p[0] for p in ideal_sounding_dots],
        y=[p[1] for p in ideal_sounding_dots],
        z=[p[2] for p in ideal_sounding_dots],
        mode='markers',
        marker=dict(color='darkblue', size=3.5, symbol='circle'),
        name='Ideal Sounding Points',
        showlegend=True
    ))

# Add 100 Actual Sounding Points (orange dots matching actual sectors)
if show_actual_soundings and len(actual_sounding_dots) > 0:
    fig.add_trace(go.Scatter3d(
        x=[p[0] for p in actual_sounding_dots],
        y=[p[1] for p in actual_sounding_dots],
        z=[p[2] for p in actual_sounding_dots],
        mode='markers',
        marker=dict(color='darkorange', size=3.5, symbol='circle'),
        name='Actual Sounding Points',
        showlegend=True
    ))


# Figure layout
fig.update_layout(
    uirevision='locked',
    height=850,
    scene=dict(
        xaxis_title='Along-Track X (m)',
        yaxis_title='Across-Track Y (m)',
        zaxis_title='Depth Z (m)',
        xaxis=dict(range=[-depth * 2.0, depth * 2.0]),

        # Invert plotly axis to match normal conventions
        yaxis=dict(range=[depth * 5.0, -depth * 5.0]),

        zaxis=dict(range=[depth * 1.1, -10]),
        aspectmode='manual',
        aspectratio=dict(x=1, y=2.5, z=0.5),

        # Initial camera perspective
        camera=dict(
            eye=dict(x=-1.5, y=0.0, z=1.0),
            center=dict(x=0.0, y=0.0, z=0.0),
            up=dict(x=0.0, y=0.0, z=-1.0)
        )
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, use_container_width=True)
