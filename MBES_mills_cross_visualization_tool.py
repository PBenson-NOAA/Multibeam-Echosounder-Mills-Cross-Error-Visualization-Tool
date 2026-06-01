import streamlit as st
import numpy as np
import plotly.graph_objects as go
from numba import njit

st.set_page_config(layout="wide", page_title="Mills Cross Error Visualization")

st.title("Multibeam Echosounder Mills Cross Visualization Tool")
st.markdown(
    "A visual aid to assess the impact of mechanical biases, dynamic IMU motion, and active beam steering on a flat seafloor baseline.")

# --- SIDEBAR INTERFACE ---
st.sidebar.header("Parameters")

# Query beam information
with st.sidebar.container(border=True):
    st.subheader("Interactive Beam Query")
    queried_angle = st.number_input("Query Specific Swath Angle (°)", min_value=-75.0, max_value=75.0, value=45.0, step=1.0)

with st.sidebar.expander("Environment", expanded=True):
    depth = st.number_input("Depth (m)", min_value=1.0, max_value=12000.0, value=100.0, step=10.0)
    c_sound = st.number_input("Sound Speed (m/s)", min_value=1400.0, max_value=1600.0, value=1500.0, step=1.0)


with st.sidebar.expander("Array Specifications", expanded=True):
    c1, c2 = st.columns(2)
    frequency = st.number_input("Frequency (Hz)", min_value=1000.0, max_value=1000000.0, value=300000.0, step=10000.0)
    tx_beamwidth = c1.number_input("TX BW (Along-Track) (°)", value=0.5, step=0.1)
    rx_beamwidth = c1.number_input("RX BW (Across-Track) (°)", value=1.0, step=0.1)
    tx_across_fan_bw = c2.number_input("TX BW (Across-Track) (°)", value=150.0, step=1.0)
    rx_fore_aft_bw = c2.number_input("RX BW (Along-Track) (°)", value=30.0, step=1.0)

    target_swath_width = st.number_input("Target Swath Coverage (°)", min_value=10.0, max_value=150.0, value=120.0,
                                         step=1.0)
    num_sectors = st.selectbox("Number of TX Sectors", options=[1, 2, 3, 4, 5, 8], index=0)
    shading_type = st.selectbox("Array Shading", options=["Uniform", "Hann", "Hamming"], index=0)


# Dynamic Motion
with st.sidebar.expander("IMU Dynamic Motion", expanded=True):
    c1, c2, c3 = st.columns(3)
    imu_roll = c1.number_input("Roll (°)", value=0.0, step=1.0)
    imu_pitch = c2.number_input("Pitch (°)", value=0.0, step=1.0)
    imu_yaw = c3.number_input("Yaw (°)", value=0.0, step=1.0)

# Static Mounting Biases
with st.sidebar.expander("Mounting Biases (Static Errors)", expanded=True):
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
with st.sidebar.expander("Active Stabilization & Steering", expanded=True):
    auto_roll = st.checkbox("Active Roll Stabilization (RX)", value=True)
    auto_pitch = st.checkbox("Active Pitch Stabilization (TX)", value=True)
    auto_yaw = st.checkbox("Active Yaw Stabilization (TX)", value=True)
    if not auto_pitch:
        manual_tx_steer = st.number_input("Manual TX Pitch Steer (°)", value=0.0, step=0.1)
    else:
        manual_tx_steer = 0.0

with st.sidebar.expander("Acoustic Lobes", expanded=True):
    show_tx_lobe = st.checkbox("Show TX Lobe (Blue)", value=False)
    if show_tx_lobe and num_sectors > 1:
        st.warning(
            "Note: The 3D TX lobe balloon does not support simultaneous multi-sector visualization. It currently renders the active queried sector only.")
    show_rx_lobe = st.checkbox("Show RX Lobe (Red)", value=False)
    show_combined_lobe = st.checkbox("Show Combined Product Lobe", value=True)

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
    bw_factor = 1.20
elif shading_type == "Hamming":
    bw_factor = 1.30

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

def generate_array_weights(N, shading="Hamming"):
    """Pre-calculates the amplitude weights for an N-element array."""
    n = np.arange(N)
    if shading == "Uniform":
        weights = np.ones(N)
    elif shading == "Hann":
        # 0.5 + 0.5 cosine weighting
        weights = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    elif shading == "Hamming":
        # 0.54 + 0.46 cosine weighting
        weights = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    else:
        weights = np.ones(N)

    # Normalize weights so the peak main-lobe amplitude is exactly 1.0
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

# --- Acoustic Directivity and Hardware Math---
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
tx_weights = generate_array_weights(comp_N_tx, shading=shading_type)
rx_weights = generate_array_weights(comp_N_rx, shading=shading_type)

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
else:
    patch_area = 0.0

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
col6.metric("Sounding Patch Area", f"{patch_area:.2f} m²")

st.markdown("---")
st.subheader("**Theoretical Array Specifications (Nominal 1500 m/s)**")

# Calculate UI variables matching the math engine
lambda_nom_ui = 1500.0 / frequency
bw_factor_ui = 0.886 if shading_type == "Uniform" else (1.20 if shading_type == "Hann" else 1.30)

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

    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            P = np.array([X_grid[i, j], Y_grid[i, j], Z_grid[i, j]])
            v_geo = P / np.linalg.norm(P)

            D_tx = calculate_directivity(v_geo, R_tx_mech, tx_steer_rad, is_tx=True)

            D_rx = calculate_directivity(v_geo, R_rx_mech, theta_rad, is_tx=False)
            I_linear = np.abs(D_tx * D_rx)
            Intensity_dB[i, j] = 20 * np.log10(I_linear + 1e-6)

    # Clip to -40 dB to clean up the visual floor
    Intensity_dB = np.clip(Intensity_dB, -40, 0)

    fig.add_trace(go.Surface(
        x=X_grid, y=Y_grid, z=Z_grid,
        surfacecolor=Intensity_dB,
        colorscale='Jet',
        cmin=-40, cmax=0,
        name='Seafloor Footprint (dB)',
        showscale=True,
        colorbar=dict(title="dB", x=0.85, len=0.5)
    ))

# --- 3D Acoustic Lobes ---
if (show_tx_lobe or show_rx_lobe or show_combined_lobe) and np.linalg.norm(pt_physical) > 0:
    # Scale lobes to extend 10% past the seafloor to clearly show the footprint overlap
    lobe_scale = np.linalg.norm(pt_physical) * 1.10


    def generate_native_lobe(is_tx, color_scale, name):
        """Generates a 3D acoustic balloon mapped to the array's native mechanical axis."""

        # Dynamically center the grid on the steered beam
        if is_tx:
            # TX array is along x steered in pitch
            v_center = np.pi / 2 - tx_steer_rad
        else:
            # RX array is along Y steered in roll
            v_center = np.pi / 2 - theta_rad

        # Both fans sweep 180 degrees downwards
        u = np.linspace(0, np.pi, 150)

        # V sweeps +/- 55 degrees around the newly centered main lobe
        # Maybe revisit this?
        v = np.linspace(v_center - np.radians(45), v_center + np.radians(45), 200)

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

        # Normalize and map to dB scale (-40dB Floor)
        R_dB = np.clip(20 * np.log10(R_linear + 1e-12), -40, 0)
        R_physical_radius = (R_dB + 40.0) / 40.0

        # Scale to push past the water column
        X_lobe = X_unit * R_physical_radius * lobe_scale
        Y_lobe = Y_unit * R_physical_radius * lobe_scale
        Z_lobe = Z_unit * R_physical_radius * lobe_scale

        return go.Surface(
            x=X_lobe, y=Y_lobe, z=Z_lobe,
            surfacecolor=R_dB,
            colorscale=color_scale,
            cmin=-40, cmax=0,
            name=name,
            showscale=False,
            opacity=0.6
        )


    # Add the Native-Aligned Individual Lobes
    if show_tx_lobe:
        fig.add_trace(generate_native_lobe(is_tx=True, color_scale='Blues', name='TX Lobe'))

    if show_rx_lobe:
        fig.add_trace(generate_native_lobe(is_tx=False, color_scale='Reds', name='RX Lobe'))

    # Add the Combined Product Lobe
    if show_combined_lobe:
        # Tightly focus a Z-down grid exactly around the intersection point
        beam_vec = pt_physical / np.linalg.norm(pt_physical)
        azimuth_center = np.arctan2(beam_vec[1], beam_vec[0])
        elevation_center = np.arccos(beam_vec[2])

        span = np.radians(8.0)
        u_comb = np.linspace(azimuth_center - span, azimuth_center + span, 80)
        v_comb = np.linspace(max(0, elevation_center - span), min(np.pi / 2, elevation_center + span), 80)
        U_comb, V_comb = np.meshgrid(u_comb, v_comb)

        X_comb = np.sin(V_comb) * np.cos(U_comb)
        Y_comb = np.sin(V_comb) * np.sin(U_comb)
        Z_comb = np.cos(V_comb)
        R_comb_linear = np.zeros_like(X_comb)

        for i in range(X_comb.shape[0]):
            for j in range(X_comb.shape[1]):
                v_geo = np.array([X_comb[i, j], Y_comb[i, j], Z_comb[i, j]])

                # Multiply TX and RX to get the combined acoustic product
                D_tx = calculate_directivity(v_geo, R_tx_mech, tx_steer_rad, is_tx=True)

                D_rx = calculate_directivity(v_geo, R_rx_mech, theta_rad, is_tx=False)
                R_comb_linear[i, j] = np.abs(D_tx * D_rx)

        R_dB = np.clip(20 * np.log10(R_comb_linear + 1e-12), -40, 0)
        R_radius = (R_dB + 40.0) / 40.0

        fig.add_trace(go.Surface(
            x=X_comb * R_radius * lobe_scale,
            y=Y_comb * R_radius * lobe_scale,
            z=Z_comb * R_radius * lobe_scale,
            surfacecolor=R_dB, colorscale='Viridis', cmin=-40, cmax=0,
            name='Combined Lobe', showscale=False, opacity=0.9
        ))

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
fig.add_trace(go.Scatter3d(x=[pt_physical[0]], y=[pt_physical[1]], z=[pt_physical[2]], mode='markers',
                           marker=dict(color='red', size=6), name='Actual Sounding'))

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
