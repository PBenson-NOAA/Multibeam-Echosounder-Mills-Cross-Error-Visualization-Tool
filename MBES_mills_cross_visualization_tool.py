import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Mills Cross Error Visualization")

st.title("Multibeam Echosounder Mills Cross Error Visualization Tool")
st.markdown(
    "A visual aid to assess the impact of mechanical biases, dynamic IMU motion, and active beam steering on a flat seafloor baseline.")

# --- SIDEBAR INTERFACE ---
st.sidebar.header("Parameters")

# Query beam information
with st.sidebar.container(border=True):
    st.subheader("Interactive Beam Query")
    queried_angle = st.number_input("Query Specific Swath Angle (°)", min_value=-75.0, max_value=75.0, value=45.0, step=1.0)

# Core System Specifications
with st.sidebar.expander("Environment & Array Specs", expanded=True):
    depth = st.number_input("Depth (m)", min_value=1.0, max_value=12000.0, value=100.0, step=10.0)
    c1, c2 = st.columns(2)
    tx_beamwidth = c1.number_input("TX BW (°)", value=0.5, step=0.1)
    rx_beamwidth = c2.number_input("RX BW (°)", value=1.0, step=0.1)
    rx_fore_aft_bw = st.number_input("RX Fore-Aft Acceptance (°)", value=30.0, step=1.0)
    tx_across_fan_bw = st.number_input("TX Across-Track Fan Limit (°)", value=150.0, step=1.0)
    num_sectors = st.selectbox("Number of TX Sectors", options=[1, 2, 3, 4, 5, 8], index=2)

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

    # This is an attempt to "intelligently" restrict transmit to region within RX listening area. Likely a gross over-simplification of what is actually done.
    # Maybe remove this since it could just add confusion?
    outer_edge_deg = 0.0
    for s_start, s_end in sector_limits:
        if s_start <= sector_center_angle <= s_end:
            outer_edge_deg = max(abs(s_start), abs(s_end))
            break

    max_allowable_steer = max(0.0, 90.0 - outer_edge_deg - (tx_beamwidth / 2.0))

    steer_deg = np.clip(steer_deg, -max_allowable_steer, max_allowable_steer)

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
tx_bw_rad = np.radians(tx_beamwidth)
rx_bw_rad = np.radians(rx_beamwidth)

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


# --- Algebraic Intersection Solving ---
def solve_mills_cross_intersection(R_tx, R_rx, tx_steer_angle_rad, rx_steer_angle_rad, seafloor_depth):
    """Algebraically solves the 3D intersection vector of a steered TX cone and a steered RX cone."""
    u_tx = R_tx[:, 0]
    u_rx = R_rx[:, 1]
    w = np.cross(u_tx, u_rx)

    cos_gamma = np.dot(u_tx, u_rx)
    D = 1.0 - cos_gamma ** 2
    if D < 1e-6:
        return np.array([0, 0, 0])

    c1 = np.sin(tx_steer_angle_rad)
    c2 = np.sin(rx_steer_angle_rad)

    a = (c1 - c2 * cos_gamma) / D
    b = (c2 - c1 * cos_gamma) / D
    inner_val = 1.0 - (a ** 2 + b ** 2 + 2 * a * b * cos_gamma)

    if inner_val < 0:
        return np.array([0, 0, 0])

    c = np.sqrt(inner_val / D)
    v1 = a * u_tx + b * u_rx + c * w
    v2 = a * u_tx + b * u_rx - c * w
    v = v1 if v1[2] > 0 else v2

    if v[2] < 1e-6:
        return np.array([0, 0, 0])

    scale = seafloor_depth / v[2]
    return v * scale


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

# Sounding patch mesh generation - algebraic method
c1 = solve_mills_cross_intersection(R_tx_mech, R_rx_mech, tx_fwd_psi, theta_min, depth)
c2 = solve_mills_cross_intersection(R_tx_mech, R_rx_mech, tx_fwd_psi, theta_max, depth)
c3 = solve_mills_cross_intersection(R_tx_mech, R_rx_mech, tx_aft_psi, theta_max, depth)
c4 = solve_mills_cross_intersection(R_tx_mech, R_rx_mech, tx_aft_psi, theta_min, depth)

patch_pts = [c1, c2, c3, c4]
has_overlap = all(np.linalg.norm(p) > 1e-3 for p in patch_pts)

if has_overlap:
    x_c = [p[0] for p in patch_pts]
    y_c = [p[1] for p in patch_pts]
    patch_area = 0.5 * np.abs(np.dot(x_c, np.roll(y_c, 1)) - np.dot(y_c, np.roll(x_c, 1)))
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
col5.metric("Target Width", f"{tx_x_width:.2f} m")
col6.metric("Sounding Patch Area", f"{patch_area:.2f} m²")

# --- GRAPH TOGGLES ---
st.markdown("---")
st.subheader("Visualization Overlays")

# condense toggles to left side of screen
t_col1, t_col2, t_col3, t_col4, t_col5, t_col6, spacer = st.columns([1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 9.0])

show_ideal_tx = t_col1.checkbox("Ideal TX Sectors", value=True)
show_actual_tx = t_col2.checkbox("Actual TX Sectors", value=True)
show_rx_bowtie = t_col3.checkbox("RX Bowtie", value=True)
show_rx_red = t_col4.checkbox("RX Footprint", value=True)
show_ideal_soundings = t_col5.checkbox("Ideal Soundings (100 beams)", value=False)
show_actual_soundings = t_col6.checkbox("Actual Soundings (100 beams)", value=False)

# --- Equidistant beam math (100 beams for simplicity) ---
ideal_sounding_dots = []
actual_sounding_dots = []

if show_ideal_soundings or show_actual_soundings:
    # Establish the global target horizontal footprint bounds at 75 degrees
    max_y = depth * np.tan(np.radians(75.0))

    # Create a uniform, linear grid of 100 target Y coordinates across the swath
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

        # Account for the conical beam projection. Dynamically scales the receive angle so the soundings form a linear grid
        sin_nominal = y_coord / np.sqrt(depth ** 2 + y_coord ** 2)
        b_rad = np.arcsin(np.clip(sin_nominal * np.cos(steer_rad), -1.0, 1.0))

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

# --- 3D VISUALIZATION ---
fig = go.Figure()

# Add Ideal TX Footprint
if show_ideal_tx:
    ideal_colors = ['blue', 'deepskyblue']
    for idx, sector_pts in enumerate(calculated_tx_sectors):
        color = ideal_colors[idx % len(ideal_colors)]
        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in sector_pts] + [sector_pts[0][0]],
            y=[p[1] for p in sector_pts] + [sector_pts[0][1]],
            z=[p[2] for p in sector_pts] + [sector_pts[0][2]],
            mode='lines', line=dict(color=color, width=2.5, dash='dash'),
            name='Ideal TX Sectors' if idx == 0 else None,
            showlegend=(idx == 0)
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

# Add RX Bowtie Boundary
if show_rx_bowtie:
    fig.add_trace(go.Scatter3d(
        x=rx_full_x, y=rx_full_y, z=rx_full_z,
        mode='lines', line=dict(color='green', width=4), name='RX Bowtie Boundary'
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
        x=[p[0] for p in patch_pts] + [patch_pts[0][0]],
        y=[p[1] for p in patch_pts] + [patch_pts[0][1]],
        z=[p[2] for p in patch_pts] + [patch_pts[0][2]],
        mode='lines', line=dict(color='purple', width=5), name='Sounding Patch'
    ))

fig.add_trace(go.Scatter3d(x=[0, pt_calculated[0]], y=[0, pt_calculated[1]], z=[0, pt_calculated[2]], mode='lines',
                           line=dict(color='blue', width=4), name='Ideal Pointing Vector'))
fig.add_trace(go.Scatter3d(x=[0, pt_physical[0]], y=[0, pt_physical[1]], z=[0, pt_physical[2]], mode='lines',
                           line=dict(color='red', width=6), name='Actual Pointing Vector'))

fig.add_trace(go.Scatter3d(x=[pt_calculated[0]], y=[pt_calculated[1]], z=[pt_calculated[2]], mode='markers',
                           marker=dict(color='blue', size=6), name='Ideal Sounding'))
fig.add_trace(go.Scatter3d(x=[pt_physical[0]], y=[pt_physical[1]], z=[pt_physical[2]], mode='markers',
                           marker=dict(color='red', size=6), name='Actual Sounding'))

# --- VISUAL ANGLE REFERENCE GRID (Dynamic Protractor following IMU Yaw) ---
ref_angles = np.arange(-75, 76, 15)  # Label every 15 degrees
lbl_x, lbl_y, lbl_z, lbl_text = [], [], [], []

# Convert IMU Yaw to radians for rotation math
grid_yaw_rad = np.radians(imu_yaw)

# Draw the transparent radial lines one-by-one to bypass WebGL bugs
for i, ang in enumerate(ref_angles):
    # Calculate the base Y position as if yaw was 0
    y_base = depth * np.tan(np.radians(ang))

    # Apply Z-axis rotation to follow IMU heading
    x_rot = -y_base * np.sin(grid_yaw_rad)
    y_rot = y_base * np.cos(grid_yaw_rad)

    fig.add_trace(go.Scatter3d(
        x=[0, x_rot], y=[0, y_rot], z=[0, depth],
        mode='lines',
        line=dict(color='gray', width=2),
        opacity=0.3,  # Apply transparency at the trace level
        name='Angle Reference Grid' if i == 0 else None,
        showlegend=False,
        legendgroup='grid',
        hoverinfo='skip'
    ))

    # Save label coordinates
    lbl_x.append(x_rot)
    lbl_y.append(y_rot)
    lbl_z.append(depth * 0.95)
    lbl_text.append(f"{ang}°")

# Add Angular Degree Labels
fig.add_trace(go.Scatter3d(
    x=lbl_x, y=lbl_y, z=lbl_z,
    mode='text', text=lbl_text, textfont=dict(color='gray', size=12),
    name='Angle Labels', hoverinfo='skip', showlegend=False
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
        yaxis=dict(range=[-depth * 5.0, depth * 5.0]),
        zaxis=dict(range=[depth * 1.1, -10]),
        aspectmode='manual',
        aspectratio=dict(x=1, y=2.5, z=0.5)
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, use_container_width=True)
