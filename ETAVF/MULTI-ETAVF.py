import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp

# ==========================================
# 0. CONFIGURATION & USER INPUT
# ==========================================
s_robot = 1.0       # Linear speed (m/s)
s_ref   = 1.0       # Reference speed
kappa   = 5.0       # Time gain
k_omega = 8.0       # Angular gain (Unicycle)
RADIO_META = 0.20   # Arrival tolerance

NUM_ROBOTS = 5      # Number of robots

print("=== MULTI-ROBOT ETA-VF SIMULATION ===")
print("Select a scenario:")
print("1: Same Start Point, Different ETAs (Staggered Arrival)")
print("2: Different Start Points, Same ETA (Synchronized Arrival)")

while True:
    try:
        val = input("Enter option (1 or 2): ")
        if val in ['1', '2']:
            SCENARIO = int(val)
            break
        print("Invalid input. Please enter 1 or 2.")
    except ValueError:
        pass

# ==========================================
# 1. FIELD MATHEMATICS
# ==========================================
def get_polar_coords(x, y):
    r = np.sqrt(x**2 + y**2)
    psi = np.abs(np.arctan2(y, -x)) 
    return r, psi

def eta_function(x, y):
    r, psi = get_polar_coords(x, y)
    if r < 1e-6: return 0.0
    if abs(psi) < 1e-4:
        factor_geom = 1.0
    else:
        factor_geom = psi / np.sin(psi)
    return (r * factor_geom) / s_ref

def gradient_eta(x, y):
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

    grad = d_eta_dr * e_r + (1/r * d_eta_dpsi) * e_psi
    return grad

# ==========================================
# 2. INITIALIZATION
# ==========================================
initial_states = [] 
target_times   = [] 

# Base start point
p0_base = [-4.0, 3.0]
theta0  = 0.0

if SCENARIO == 1:
    # --- SCENARIO 1: SAME START, DIFFERENT ETAs ---
    # Use curved path time for accuracy
    t_min_phys = eta_function(p0_base[0], p0_base[1])
    print(f"Minimum physical time from base (curved): {t_min_phys:.2f}s")
    
    # Base ETA with buffer
    base_eta = np.ceil(t_min_phys) + 5.0 
    
    for i in range(NUM_ROBOTS):
        initial_states.append(p0_base + [theta0])
        eta = base_eta + (i * 4.0) 
        target_times.append(eta)

elif SCENARIO == 2:
    # --- SCENARIO 2: DIFFERENT STARTS, SAME ETA ---
    starts = [
        [-4.0, 3.0],
        [-2.0, 5.0],
        [-5.0, 2.0],
        [-7.0, 5.0], 
        [-3.0, 1.0]
    ]
    
    # Calculate max required time using the CORRECT eta_function
    max_t_needed = 0
    for p in starts:
        t_curve = eta_function(p[0], p[1])
        if t_curve > max_t_needed: 
            max_t_needed = t_curve
        initial_states.append(list(p) + [theta0])
        
    print(f"Worst-case path requires: {max_t_needed:.2f}s")
    
    # Common ETA with a safe buffer
    common_eta = np.ceil(max_t_needed) + 5.0
    target_times = [common_eta] * NUM_ROBOTS

print("Assigned ETAs:", target_times)
y0_flat = np.concatenate(initial_states)

# ==========================================
# 3. DYNAMICS & CONTROL (WITH STATION KEEPING)
# ==========================================
def compute_control_single(t, x, y, theta, T_target):
    dist = np.sqrt(x**2 + y**2)
    
    # 1. STRICT STOP CONDITION
    if dist < RADIO_META and t >= T_target:
        return 0.0, 0.0, 0.0 
    
    # 2. STATION KEEPING (The "Don't Loop Out" Fix)
    # If inside tolerance but EARLY, just orbit locally to wait.
    if dist < RADIO_META:
        # Determine best tangent direction (maintain current rotation)
        ur = np.array([x, y]) / (dist + 1e-6)
        t1 = np.array([-ur[1], ur[0]]) # Tangent 1
        t2 = np.array([ur[1], -ur[0]]) # Tangent 2
        
        heading = np.array([np.cos(theta), np.sin(theta)])
        
        # Pick tangent closest to current heading
        if np.dot(heading, t1) > np.dot(heading, t2):
            u_ideal = s_robot * t1
        else:
            u_ideal = s_robot * t2
            
        # Optional: Slight pull to center if getting too close to edge
        if dist > 0.8 * RADIO_META:
             u_ideal = u_ideal - 0.5 * s_robot * ur
             u_ideal = u_ideal / np.linalg.norm(u_ideal) * s_robot

    else:
        # 3. STANDARD ETA-VF LOGIC (Outside tolerance)
        eta_act = eta_function(x, y)
        t_offset = RADIO_META / s_robot
        T_controller = T_target + t_offset
        
        eta_ref = T_controller - t
        e = eta_act - eta_ref
        
        grad = gradient_eta(x, y)
        norm_grad = np.linalg.norm(grad)
        
        denom = s_robot * norm_grad
        if denom < 1e-6: denom = 1e-6
        
        cos_beta = (1.0 + (kappa/2.0)*e) / denom
        cos_beta = np.clip(cos_beta, -1.0, 1.0)
        sin_beta = np.sqrt(1 - cos_beta**2)
        
        if norm_grad > 1e-6:
            n = -grad / norm_grad 
            tau = np.array([-n[1], n[0]])
            if y < 0: tau = -tau 
        else:
            n = np.array([-1, 0])
            tau = np.array([0, 1])

        u_ideal = s_robot * (cos_beta * n + sin_beta * tau)
    
    # 4. UNICYCLE TRACKING
    theta_star = np.arctan2(u_ideal[1], u_ideal[0])
    heading_error = np.arctan2(np.sin(theta - theta_star), np.cos(theta - theta_star))
    omega = -k_omega * np.sin(heading_error)
    
    return s_robot * np.cos(theta), s_robot * np.sin(theta), omega

def dynamics_multi(t, state_flat):
    dydt = []
    for i in range(NUM_ROBOTS):
        idx = i * 3
        xi, yi, thi = state_flat[idx], state_flat[idx+1], state_flat[idx+2]
        Ti = target_times[i]
        
        dx, dy, dth = compute_control_single(t, xi, yi, thi, Ti)
        dydt.extend([dx, dy, dth])
    return dydt

# ==========================================
# 4. SIMULATION
# ==========================================
max_time = max(target_times) + 3.0
t_span = (0, max_time)

print("Simulating...")
sol = solve_ivp(dynamics_multi, t_span, y0_flat, max_step=0.05, rtol=1e-5)

t_anim = np.linspace(0, max_time, 600)
sol_anim = []
for i in range(len(y0_flat)):
    sol_anim.append(np.interp(t_anim, sol.t, sol.y[i]))
sol_anim = np.array(sol_anim)

# Pre-calculate errors for plotting
all_errors = []
for r_idx in range(NUM_ROBOTS):
    r_errors = []
    idx_base = r_idx * 3
    T_tgt = target_times[r_idx]
    
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
            eta_val = eta_function(x, y)
            t_offset = RADIO_META / s_robot
            e = eta_val - ((T_tgt + t_offset) - t)
        r_errors.append(e)
    all_errors.append(np.array(r_errors))

# ==========================================
# 5. VISUALIZATION
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- PLOT 1: TRAJECTORIES ---
ax1.set_xlim(-8, 2)
ax1.set_ylim(-2, 8)
ax1.grid(True, alpha=0.3)
ax1.set_title(f"Trajectory (Scenario {SCENARIO})")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

circle = plt.Circle((0, 0), RADIO_META, color='r', fill=True, alpha=0.3, label='Target')
ax1.add_patch(circle)

xx = np.linspace(-8, 2, 25)
yy = np.linspace(-2, 8, 25)
Xg, Yg = np.meshgrid(xx, yy)
Ug = Xg**2 - Yg**2
Vg = 2*Xg*Yg
Mg = np.sqrt(Ug**2 + Vg**2); Mg[Mg==0]=1
ax1.quiver(Xg, Yg, Ug/Mg, Vg/Mg, color='lightgray', scale=30, alpha=0.5)

# --- PLOT 2: TIME ERROR ---
ax2.set_xlim(0, max_time)
flat_errors = np.concatenate(all_errors)
max_e = max(np.max(flat_errors), 1.0)
min_e = min(np.min(flat_errors), -1.0)
ax2.set_ylim(min_e*1.1, max_e*1.1)
ax2.grid(True, alpha=0.3)
ax2.set_title("Time Error")
ax2.axhline(0, c='k', ls=':', alpha=0.5)
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Error [s]")

colors = plt.cm.jet(np.linspace(0, 1, NUM_ROBOTS))

trails = []
robots = []
quivers_robot = []
error_lines = []
error_dots = []

for i in range(NUM_ROBOTS):
    c = colors[i]
    idx_base = i * 3
    
    # Initialize Visuals
    tr, = ax1.plot([], [], '-', color=c, lw=1.5)
    rob, = ax1.plot([], [], 'o', color=c, ms=6)
    
    # Correct Quiver Initialization
    init_x, init_y, init_th = initial_states[i]
    quiv = ax1.quiver([init_x], [init_y], [np.cos(init_th)], [np.sin(init_th)], 
                      color='k', scale=20, width=0.005)
    
    trails.append(tr)
    robots.append(rob)
    quivers_robot.append(quiv)
    
    el, = ax2.plot([], [], '-', color=c, lw=1.5, label=f'R{i+1}')
    ed, = ax2.plot([], [], 'o', color='k', ms=4)
    ax2.axvline(target_times[i], color=c, ls='--', alpha=0.3)
    
    error_lines.append(el)
    error_dots.append(ed)

ax2.legend(loc='upper right', fontsize='small')

time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, va='top', ha='left',
                     bbox=dict(boxstyle="round", facecolor='white', alpha=0.9, edgecolor='gray'))

def init():
    for i in range(NUM_ROBOTS):
        trails[i].set_data([], [])
        robots[i].set_data([], [])
        error_lines[i].set_data([], [])
        error_dots[i].set_data([], [])
        # Do not clear quivers to avoid reset bug
    time_text.set_text('')
    return trails + robots + quivers_robot + error_lines + error_dots + [time_text]

def update(frame):
    current_time = t_anim[frame]
    info_str = f"Time: {current_time:.2f} s\nScenario: {SCENARIO}"
    
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

step = 3
frames = range(0, len(t_anim), step)

print("Generating animation (Saving 'multi_robot.gif')...")
ani = FuncAnimation(fig, update, frames=frames, init_func=init, interval=20, blit=False)

try:
    ani.save("multi_robot.gif", writer=PillowWriter(fps=30))
    print("Done! Animation saved.")
except Exception as e:
    print(f"Error saving GIF: {e}")

plt.show()