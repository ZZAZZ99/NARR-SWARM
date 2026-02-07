import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp

# ==========================================
# 0. CONFIGURATION & USER INTERFACE
# ==========================================
s_ref   = 1.0       # Reference speed
kappa   = 5.0       # Time gain
k_omega = 8.0       # Angular gain
RADIO_META = 0.20   # Arrival tolerance
NUM_ROBOTS = 5      # Number of robots

# Base Noise Parameters
BASE_NOISE_MAG = 0.2
NOISE_FREQ     = 2.0

print("=== MULTI-ROBOT ETA-VF SIMULATION (FIXED) ===")

# --- 1. SPEED CONFIGURATION ---
print("\n[1] SPEED CONFIGURATION")
print("1: Constant Speed (Ideal)")
print("2: Noisy Speed (Robustness Test)")

NOISE_MAGNITUDE = 0.0

while True:
    val = input("Select Speed Mode (1 or 2): ")
    if val == '1':
        NOISE_MAGNITUDE = 0.0
        print(">> CONSTANT (No noise)")
        break
    elif val == '2':
        NOISE_MAGNITUDE = BASE_NOISE_MAG
        print(f">> NOISY (Fluctuation +/- {NOISE_MAGNITUDE} m/s)")
        break

# --- 2. SCENARIO SELECTION ---
print("\n[2] SCENARIO SELECTION")
print("1: Same Start, Different ETAs, Same Speed")
print("2: Different Starts, Same ETA, Same Speed")
print("3: Same Start, Same ETA, DIFFERENT SPEEDS")
print("4: Different Starts, Same ETA, DIFFERENT SPEEDS")

while True:
    try:
        val = input("Select Scenario (1-4): ")
        if val in ['1', '2', '3', '4']:
            SCENARIO = int(val)
            break
    except ValueError:
        pass

# --- 3. GIF OUTPUT ---
print("\n[3] OUTPUT")
SAVE_GIF = False
while True:
    val = input("Generate and save GIF? (y/n): ").lower()
    if val == 'y':
        SAVE_GIF = True
        break
    elif val == 'n':
        break

# ==========================================
# 1. FIELD MATHEMATICS
# ==========================================
def get_polar_coords(x, y):
    r = np.sqrt(x**2 + y**2)
    psi = np.abs(np.arctan2(y, -x)) 
    return r, psi

def eta_function_ref(x, y):
    r, psi = get_polar_coords(x, y)
    if r < 1e-6: return 0.0
    if abs(psi) < 1e-4: factor = 1.0
    else:               factor = psi / np.sin(psi)
    return (r * factor) / s_ref

def gradient_eta_ref(x, y):
    r, psi = get_polar_coords(x, y)
    if r < 1e-6: return np.array([-1.0, 0.0])
    e_r = np.array([x/r, y/r])
    if y >= 0: vec_tan = np.array([y, -x]) 
    else:      vec_tan = np.array([-y, x])
    e_psi = vec_tan / np.linalg.norm(vec_tan)

    if abs(psi) < 1e-4:
        d_eta_dr = 1.0 / s_ref
        d_eta_dpsi = 0.0
    else:
        d_eta_dr = (psi / np.sin(psi)) / s_ref
        d_eta_dpsi = (r / s_ref) * (np.sin(psi) - psi*np.cos(psi)) / (np.sin(psi)**2)

    return d_eta_dr * e_r + (1/r * d_eta_dpsi) * e_psi

# ==========================================
# 2. SCENARIO SETUP
# ==========================================
initial_states = [] 
target_times   = []
robot_nominal_speeds = [] 

p0_base = [-4.0, 3.0]
theta0  = 0.0

if SCENARIO == 1:
    robot_nominal_speeds = [1.0] * NUM_ROBOTS
    t_base = eta_function_ref(p0_base[0], p0_base[1]) * (s_ref/1.0)
    print(f"\nMin Physical Time: {t_base:.2f}s")
    base_eta = np.ceil(t_base) + 5.0
    for i in range(NUM_ROBOTS):
        initial_states.append(p0_base + [theta0])
        target_times.append(base_eta + i*4.0)

elif SCENARIO == 2:
    robot_nominal_speeds = [1.0] * NUM_ROBOTS
    starts = [[-4.0, 3.0], [-2.0, 5.0], [-5.0, 2.0], [-7.0, 5.0], [-3.0, 1.0]]
    max_t = 0
    for p in starts:
        t = eta_function_ref(p[0], p[1]) * (s_ref/1.0)
        if t > max_t: max_t = t
        initial_states.append(list(p) + [theta0])
    print(f"\nMax Required Time: {max_t:.2f}s")
    target_times = [np.ceil(max_t) + 5.0] * NUM_ROBOTS

elif SCENARIO == 3:
    # Critical case where the delay happened
    robot_nominal_speeds = [0.6, 0.8, 1.0, 1.2, 1.4]
    eta_ref = eta_function_ref(p0_base[0], p0_base[1])
    # The slowest robot (0.6) dictates the ETA
    t_min = eta_ref * (s_ref / min(robot_nominal_speeds))
    print(f"\nSlowest robot requires: {t_min:.2f}s")
    common_eta = np.ceil(t_min) + 5.0
    for i in range(NUM_ROBOTS):
        initial_states.append(p0_base + [theta0])
        target_times.append(common_eta)

elif SCENARIO == 4:
    starts = [[-4.0, 3.0], [-2.0, 5.0], [-5.0, 2.0], [-7.0, 5.0], [-3.0, 1.0]]
    robot_nominal_speeds = [1.2, 0.8, 1.5, 0.6, 1.0]
    max_t = 0
    for i in range(NUM_ROBOTS):
        t = eta_function_ref(starts[i][0], starts[i][1]) * (s_ref/robot_nominal_speeds[i])
        if t > max_t: max_t = t
        initial_states.append(list(starts[i]) + [theta0])
    print(f"\nWorst-case requires: {max_t:.2f}s")
    target_times = [np.ceil(max_t) + 5.0] * NUM_ROBOTS

y0_flat = np.concatenate(initial_states)

# ==========================================
# 3. DYNAMICS & CONTROL (FIXED)
# ==========================================
def compute_control_single(t, x, y, theta, T_target, s_nominal, s_actual):
    dist = np.sqrt(x**2 + y**2)
    
    # 1. STOP CONDITION (Strict)
    if dist < RADIO_META and t >= T_target:
        return 0.0, 0.0, 0.0 
    
    # 2. STATION KEEPING (FIXED FOR NOISE ROBUSTNESS)
    if dist < RADIO_META:
        ur = np.array([x, y]) / (dist + 1e-6)
        t1 = np.array([-ur[1], ur[0]]) 
        t2 = np.array([ur[1], -ur[0]]) 
        heading = np.array([np.cos(theta), np.sin(theta)])
        
        # Choose tangential direction
        if np.dot(heading, t1) > np.dot(heading, t2):
            u_dir = t1
        else:
            u_dir = t2
        
        # --- FIX IS HERE ---
        # Previous logic: 0.8 * RADIO_META (Too close to edge, noise pushes robot out)
        # New logic:      0.4 * RADIO_META (Keeps robot deep inside the safe zone)
        if dist > 0.4 * RADIO_META: 
             # Strong centering force
             u_dir = u_dir - 0.8 * ur 
             u_dir = u_dir / np.linalg.norm(u_dir)
             
        u_actual = s_actual * u_dir

    else:
        # 3. STANDARD ETA-VF LOGIC
        eta_map = eta_function_ref(x, y)
        eta_est = eta_map * (s_ref / s_nominal) 
        
        grad_map = gradient_eta_ref(x, y)
        grad_est = grad_map * (s_ref / s_nominal)
        norm_grad_est = np.linalg.norm(grad_est)
        
        t_offset = RADIO_META / s_nominal
        e = eta_est - ((T_target + t_offset) - t)
        
        denom = s_nominal * norm_grad_est
        if denom < 1e-6: denom = 1e-6
        
        cos_beta = (1.0 + (kappa/2.0)*e) / denom
        cos_beta = np.clip(cos_beta, -1.0, 1.0)
        sin_beta = np.sqrt(1 - cos_beta**2)
        
        if norm_grad_est > 1e-6:
            n = -grad_est / norm_grad_est
            tau = np.array([-n[1], n[0]])
            if y < 0: tau = -tau 
        else:
            n = np.array([-1, 0])
            tau = np.array([0, 1])

        dir_ideal = (cos_beta * n + sin_beta * tau)
        u_actual = s_actual * dir_ideal
    
    # 4. UNICYCLE CONTROL
    theta_star = np.arctan2(u_actual[1], u_actual[0])
    heading_error = np.arctan2(np.sin(theta - theta_star), np.cos(theta - theta_star))
    omega = -k_omega * np.sin(heading_error)
    
    return s_actual * np.cos(theta), s_actual * np.sin(theta), omega

def dynamics_multi(t, state_flat):
    dydt = []
    for i in range(NUM_ROBOTS):
        idx = i * 3
        xi, yi, thi = state_flat[idx], state_flat[idx+1], state_flat[idx+2]
        Ti = target_times[i]
        s_nom = robot_nominal_speeds[i]
        
        # APPLY NOISE
        phase = i * (np.pi/2) 
        noise = NOISE_MAGNITUDE * np.sin(NOISE_FREQ * t + phase)
        s_phys = s_nom + noise
        if s_phys < 0.1: s_phys = 0.1 
        
        dx, dy, dth = compute_control_single(t, xi, yi, thi, Ti, s_nom, s_phys)
        dydt.extend([dx, dy, dth])
    return dydt

# ==========================================
# 4. SIMULATION
# ==========================================
max_time = max(target_times) + 3.0
t_span = (0, max_time)

print(f"\nSimulating (Noise={NOISE_MAGNITUDE})...")
sol = solve_ivp(dynamics_multi, t_span, y0_flat, max_step=0.05, rtol=1e-5)

t_anim = np.linspace(0, max_time, 600)
sol_anim = []
for i in range(len(y0_flat)):
    sol_anim.append(np.interp(t_anim, sol.t, sol.y[i]))
sol_anim = np.array(sol_anim)

# Error Calculation
all_errors = []
for r_idx in range(NUM_ROBOTS):
    r_errors = []
    idx_base = r_idx * 3
    T_tgt = target_times[r_idx]
    s_curr = robot_nominal_speeds[r_idx] 
    
    arrival_flag = False
    
    for i, t in enumerate(t_anim):
        x = sol_anim[idx_base, i]
        y = sol_anim[idx_base+1, i]
        dist = np.sqrt(x**2 + y**2)
        
        if not arrival_flag:
            if dist < RADIO_META and t >= T_tgt:
                arrival_flag = True
        
        if arrival_flag:
            e = 0.0
        else:
            eta_map = eta_function_ref(x, y)
            eta_est = eta_map * (s_ref / s_curr)
            t_offset = RADIO_META / s_curr
            e = eta_est - ((T_tgt + t_offset) - t)
        r_errors.append(e)
    all_errors.append(np.array(r_errors))

# ==========================================
# 5. VISUALIZATION
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

mode_str = "CONST" if NOISE_MAGNITUDE == 0 else "NOISY"
ax1.set_xlim(-8, 2)
ax1.set_ylim(-2, 8)
ax1.grid(True, alpha=0.3)
ax1.set_title(f"Trajectory (Scenario {SCENARIO}) - {mode_str}")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

circle = plt.Circle((0, 0), RADIO_META, color='r', fill=True, alpha=0.3, label='Goal')
ax1.add_patch(circle)

xx = np.linspace(-8, 2, 25)
yy = np.linspace(-2, 8, 25)
Xg, Yg = np.meshgrid(xx, yy)
Ug = Xg**2 - Yg**2
Vg = 2*Xg*Yg
Mg = np.sqrt(Ug**2 + Vg**2); Mg[Mg==0]=1
ax1.quiver(Xg, Yg, Ug/Mg, Vg/Mg, color='lightgray', scale=30, alpha=0.5)

ax2.set_xlim(0, max_time)
flat_errors = np.concatenate(all_errors)
max_e = max(np.max(flat_errors), 1.0)
min_e = min(np.min(flat_errors), -1.0)
ax2.set_ylim(min_e*1.1, max_e*1.1)
ax2.grid(True, alpha=0.3)
ax2.set_title(f"Time Error ({mode_str})")
ax2.axhline(0, c='k', ls=':', alpha=0.5)
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Error [s]")

colors = plt.cm.jet(np.linspace(0, 1, NUM_ROBOTS))

trails, robots, quivers_robot, error_lines, error_dots = [], [], [], [], []

for i in range(NUM_ROBOTS):
    c = colors[i]
    v = robot_nominal_speeds[i]
    lbl = f"R{i+1} (v~{v:.1f})"
    
    init_x, init_y, init_th = initial_states[i]
    tr, = ax1.plot([], [], '-', color=c, lw=1.5, label=lbl)
    rob, = ax1.plot([], [], 'o', color=c, ms=6)
    quiv = ax1.quiver([init_x], [init_y], [np.cos(init_th)], [np.sin(init_th)], 
                      color='k', scale=20, width=0.005)
    
    trails.append(tr)
    robots.append(rob)
    quivers_robot.append(quiv)
    
    el, = ax2.plot([], [], '-', color=c, lw=1.5)
    ed, = ax2.plot([], [], 'o', color='k', ms=4)
    ax2.axvline(target_times[i], color=c, ls='--', alpha=0.3)
    
    error_lines.append(el)
    error_dots.append(ed)

ax1.legend(loc='upper right', fontsize='small')
time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, va='top', ha='left',
                     bbox=dict(boxstyle="round", facecolor='white', alpha=0.9, edgecolor='gray'))

def init():
    for i in range(NUM_ROBOTS):
        trails[i].set_data([], [])
        robots[i].set_data([], [])
        error_lines[i].set_data([], [])
        error_dots[i].set_data([], [])
    time_text.set_text('')
    return trails + robots + quivers_robot + error_lines + error_dots + [time_text]

def update(frame):
    current_time = t_anim[frame]
    info_str = f"T: {current_time:.2f}s\nMode: {mode_str}"
    
    for i in range(NUM_ROBOTS):
        idx_base = i * 3
        
        xs = sol_anim[idx_base, :frame+1]
        ys = sol_anim[idx_base+1, :frame+1]
        x_curr = sol_anim[idx_base, frame]
        y_curr = sol_anim[idx_base+1, frame]
        th_curr = sol_anim[idx_base+2, frame]
        
        trails[i].set_data(xs, ys)
        robots[i].set_data([x_curr], [y_curr])
        quivers_robot[i].set_offsets([x_curr, y_curr])
        quivers_robot[i].set_UVC([np.cos(th_curr)], [np.sin(th_curr)])
        
        es = all_errors[i][:frame+1]
        e_curr = all_errors[i][frame]
        
        error_lines[i].set_data(t_anim[:frame+1], es)
        error_dots[i].set_data([current_time], [e_curr])
        
    time_text.set_text(info_str)
    return trails + robots + quivers_robot + error_lines + error_dots + [time_text]

ani = FuncAnimation(fig, update, frames=range(0, len(t_anim), 3), init_func=init, interval=20, blit=False)

if SAVE_GIF:
    print("Saving GIF...")
    try:
        filename = "sim_robot_noisy.gif" if NOISE_MAGNITUDE > 0 else "sim_robot_const.gif"
        ani.save(filename, writer=PillowWriter(fps=30))
        print(f"Saved as '{filename}'.")
    except Exception as e:
        print(f"Error saving GIF: {e}")

plt.show()