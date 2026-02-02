import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp
import sys

# ==========================================
# 0. DYNAMICS SELECTION
# ==========================================
print("=== DYNAMICS SELECTION ===")
print("1: Particle (Pure kinematics, v=const)")
print("2: Unicycle (Non-holonomic, v=const, controls omega)")
while True:
    try:
        mode_input = input("Choose option (1 or 2): ")
        if mode_input in ['1', '2']:
            MODE = int(mode_input)
            break
    except:
        pass

# ==========================================
# 1. PARAMETERS
# ==========================================
s_robot = 1.0       
s_ref   = 1.0       
kappa   = 5.0       
k_omega = 8.0       # Gain for Unicycle
p0_xy   = [-4.0, 3.0] 
theta0  = 0.0       

# --- RADIUS CONFIGURATION ---
# Mode 1: High precision (5cm)
# Mode 2: Physical tolerance (20cm)
# But BOTH will now strictly respect the time constraint.
if MODE == 1:
    RADIO_META = 0.05   
    print(f"Mode 1 Selected: Radius {RADIO_META}m")
else:
    RADIO_META = 0.20   
    print(f"Mode 2 Selected: Radius {RADIO_META}m")

# Initial state
if MODE == 1:
    y0 = p0_xy
else:
    y0 = p0_xy + [theta0]

# ==========================================
# 2. FIELD MATHEMATICS (Original)
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
# 3. INTERACTION
# ==========================================
dist_ini = np.linalg.norm(p0_xy)
t_min_fisico = (dist_ini - RADIO_META) / s_robot 

print(f"\nPhysical minimum: {t_min_fisico:.4f} s")

while True:
    try:
        val = input(f"Enter desired ETA (> {t_min_fisico:.2f}): ")
        T_user = float(val)
        if T_user >= t_min_fisico: break
        print("ETA too small.")
    except:
        pass

t_offset = RADIO_META / s_robot
T_controlador = T_user + t_offset

# ==========================================
# 4. DYNAMICS AND CONTROL
# ==========================================
def compute_guidance_vector(t, x, y):
    """
    Original Control Logic.
    """
    eta_act = eta_function(x, y)
    eta_ref = T_controlador - t
    e = eta_act - eta_ref
    
    grad = gradient_eta(x, y)
    norm_grad = np.linalg.norm(grad)
    
    denom = s_robot * norm_grad
    if denom < 1e-6: denom = 1e-6
    
    # Control Law
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
    return u_ideal, e

def dynamics(t, state):
    x = state[0]
    y = state[1]
    dist = np.sqrt(x**2 + y**2)
    
    # --- STRICT STOP CONDITION FOR BOTH ---
    # Stops ONLY if inside radius AND time has passed.
    if dist < RADIO_META and t >= T_user:
        if MODE == 1: return [0.0, 0.0]
        else:         return [0.0, 0.0, 0.0]

    u_ideal, _ = compute_guidance_vector(t, x, y)
    
    if MODE == 1:
        return list(u_ideal)
    else:
        theta = state[2]
        theta_star = np.arctan2(u_ideal[1], u_ideal[0])
        heading_error = theta - theta_star
        omega = -k_omega * np.sin(heading_error)
        return [s_robot * np.cos(theta), s_robot * np.sin(theta), omega]

# ==========================================
# 5. SOLVER
# ==========================================
print(f"Simulating mode {MODE}...")
t_span = (0, T_user * 1.5) 
sol = solve_ivp(dynamics, t_span, y0, max_step=0.01, dense_output=True)

t_anim = np.linspace(0, t_span[1], 600)
traj = sol.sol(t_anim)
X_traj = traj[0]
Y_traj = traj[1]

# ==========================================
# 6. POST-PROCESS
# ==========================================
errors = []
arrival_idx = -1

for i, t in enumerate(t_anim):
    p = np.array([X_traj[i], Y_traj[i]])
    dist = np.linalg.norm(p)
    
    # Strict arrival detection for BOTH modes
    if arrival_idx == -1 and dist < RADIO_META and t >= T_user:
        arrival_idx = i
    
    if arrival_idx != -1 and i >= arrival_idx:
        e = 0.0
    else:
        eta = eta_function(p[0], p[1])
        e = eta - (T_controlador - t)
    errors.append(e)

real_arrival_time = t_anim[arrival_idx] if arrival_idx != -1 else t_anim[-1]
print(f"Actual Arrival: {real_arrival_time:.4f} s")
print(f"Target Time:    {T_user:.4f} s")
print(f"Difference:     {real_arrival_time - T_user:.4f} s")

# ==========================================
# 7. VISUALIZATION
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.set_xlim(-6, 1)
ax1.set_ylim(-5, 5)
ax1.grid(True, alpha=0.3)
title_mode = "PARTICLE" if MODE == 1 else "UNICYCLE"
ax1.set_title(f"Trajectory ({title_mode}) - Target: {T_user}s")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

circle = plt.Circle((0, 0), RADIO_META, color='r', fill=True, alpha=0.3, label='Target')
ax1.add_patch(circle)

# Background Field
xx = np.linspace(-6, 1, 20)
yy = np.linspace(-5, 5, 20)
Xg, Yg = np.meshgrid(xx, yy)
Ug = Xg**2 - Yg**2
Vg = 2*Xg*Yg
Mg = np.sqrt(Ug**2 + Vg**2); Mg[Mg==0]=1
ax1.quiver(Xg, Yg, Ug/Mg, Vg/Mg, color='lightgray', scale=30, alpha=0.5)

trail, = ax1.plot([], [], 'b-', lw=1.5)
robot, = ax1.plot([], [], 'bo', ms=6)
quiver_robot = ax1.quiver([], [], [], [], color='k', scale=20, width=0.01)
time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, va='top', ha='left',
                     bbox=dict(boxstyle="round", facecolor='white', alpha=0.9, edgecolor='gray'))

# Error Plot
ax2.set_xlim(0, t_span[1])
max_e = max(np.max(errors), 0.5)
min_e = min(np.min(errors), -0.5)
ax2.set_ylim(min_e*1.2, max_e*1.2)
ax2.grid(True, alpha=0.3)
ax2.set_title("Time Error")
ax2.axvline(T_user, c='g', ls='--', alpha=0.5)
ax2.axhline(0, c='k', ls=':', alpha=0.5)
err_line, = ax2.plot([], [], 'r-', lw=1.5)
err_dot, = ax2.plot([], [], 'ko', ms=4)

def init():
    trail.set_data([], [])
    robot.set_data([], [])
    err_line.set_data([], [])
    err_dot.set_data([], [])
    quiver_robot.set_UVC([], [])
    return trail, robot, err_line, err_dot, quiver_robot

def update(frame):
    if arrival_idx != -1 and frame >= arrival_idx:
        idx = arrival_idx
        display_t = real_arrival_time
        status = "ARRIVED"
        col = 'green'
    else:
        idx = frame
        display_t = t_anim[frame]
        status = "NAVIGATING"
        col = 'blue' if MODE==1 else 'purple'
        if display_t > T_user + 0.05: # Slight tolerance for "late" label
             status = "DELAYED"
             col = 'red'
        
    trail.set_data(X_traj[:idx+1], Y_traj[:idx+1])
    robot.set_data([X_traj[idx]], [Y_traj[idx]])
    
    if MODE == 2:
        th = traj[2][idx]
        quiver_robot.set_offsets([X_traj[idx], Y_traj[idx]])
        quiver_robot.set_UVC([np.cos(th)], [np.sin(th)])
    
    err_line.set_data(t_anim[:frame+1], errors[:frame+1])
    err_dot.set_data([display_t], [errors[idx]])
    
    info = f"Mode: {title_mode}\nTgt: {T_user:.2f}s\nSim: {display_t:.2f}s\n{status}"
    time_text.set_text(info)
    time_text.get_bbox_patch().set_edgecolor(col)
    
    return trail, robot, err_line, err_dot, quiver_robot, time_text

step = 2
frames = range(0, len(t_anim), step)
ani = FuncAnimation(fig, update, frames=frames, init_func=init, interval=20, blit=False)

print("Saving GIF 'trajectory.gif'...")
try:
    ani.save("trajectory.gif", writer=PillowWriter(fps=30))
    print("Done!")
except Exception as e:
    print(f"Error: {e}")

plt.show()