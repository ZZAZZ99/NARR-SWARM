import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp
import sys

# ==========================================
# 1. PARÁMETROS
# ==========================================
s_robot = 1.0       
s_ref   = 1.0       
kappa   = 5.0       
p0 = [-4.0, 3.0]    
RADIO_META = 0.05 

# ==========================================
# 2. MATEMÁTICA CONTINUA (Sin saltos)
# ==========================================
def get_polar_coords(x, y):
    r = np.sqrt(x**2 + y**2)
    # Ángulo en el semiplano izquierdo/estable
    psi = np.abs(np.arctan2(y, -x)) 
    return r, psi

def eta_function(x, y):
    r, psi = get_polar_coords(x, y)
    
    # Límite matemático cuando r -> 0 es 0.
    if r < 1e-6: return 0.0
    
    # Límite cuando psi -> 0 (Eje X) es 1.
    # Usamos sinc (sin(x)/x) para estabilidad numérica
    # eta = r * (psi / sin(psi)) / s
    # Si psi es muy pequeño, psi/sin(psi) -> 1
    if abs(psi) < 1e-4:
        factor_geom = 1.0
    else:
        factor_geom = psi / np.sin(psi)
        
    return (r * factor_geom) / s_ref

def gradient_eta(x, y):
    r, psi = get_polar_coords(x, y)
    
    # Singularidad estricta en el origen
    if r < 1e-6: return np.array([-1.0, 0.0])
    
    e_r = np.array([x/r, y/r])
    if y >= 0: vec_tan = np.array([y, -x]) 
    else:      vec_tan = np.array([-y, x])
    e_psi = vec_tan / np.linalg.norm(vec_tan)

    # Derivadas continuas
    if abs(psi) < 1e-4:
        d_eta_dr = 1.0 / s_ref
        d_eta_dpsi = 0.0
    else:
        d_eta_dr = (psi / np.sin(psi)) / s_ref
        d_eta_dpsi = (r / s_ref) * (np.sin(psi) - psi*np.cos(psi)) / (np.sin(psi)**2)

    grad = d_eta_dr * e_r + (1/r * d_eta_dpsi) * e_psi
    return grad

# ==========================================
# 3. INTERACCIÓN
# ==========================================
dist_ini = np.linalg.norm(p0)
t_min_fisico = (dist_ini - RADIO_META) / s_robot 

print(f"Mínimo físico: {t_min_fisico:.4f} s")

while True:
    try:
        val = input(f"Introduce ETA deseado (> {t_min_fisico:.2f}): ")
        T_user = float(val)
        if T_user >= t_min_fisico: break
        print("Imposible.")
    except:
        pass

# Compensación
t_offset = RADIO_META / s_robot
T_controlador = T_user + t_offset

# ==========================================
# 4. DINÁMICA
# ==========================================
def control_law(t, p):
    x, y = p
    dist = np.linalg.norm(p)
    
    # Condición de Parada
    if dist < RADIO_META and t >= T_user:
        return [0.0, 0.0]
        
    grad = gradient_eta(x, y)
    norm_grad = np.linalg.norm(grad)
    
    eta_act = eta_function(x, y)
    eta_ref = T_controlador - t
    e = eta_act - eta_ref
    
    denom = s_robot * norm_grad
    if denom < 1e-6: denom = 1e-6
    
    cos_beta = (1.0 + (kappa/2.0)*e) / denom
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    sin_beta = np.sqrt(1 - cos_beta**2)
    
    n = -grad / norm_grad 
    tau = np.array([-n[1], n[0]]) 
    if y < 0: tau = -tau
    
    u = s_robot * (cos_beta * n + sin_beta * tau)
    return u

def dynamics(t, p):
    return control_law(t, p)

# ==========================================
# 5. SOLVER
# ==========================================
t_span = (0, T_user * 1.3)
sol = solve_ivp(dynamics, t_span, p0, max_step=0.01, dense_output=True)

t_anim = np.linspace(0, t_span[1], 500)
pos = sol.sol(t_anim)
X_traj, Y_traj = pos

# ==========================================
# 6. VISUALIZACIÓN
# ==========================================
# Detectar llegada
arrival_idx = -1
for i, t in enumerate(t_anim):
    p = pos[:, i]
    if np.linalg.norm(p) < RADIO_META and t >= T_user:
        arrival_idx = i
        break

real_arrival_time = t_anim[arrival_idx] if arrival_idx != -1 else t_anim[-1]
print(f"Llegada Real: {real_arrival_time:.4f} s")

# Calcular errores (limpios)
errors = []
for i, t in enumerate(t_anim):
    p = pos[:, i]
    # Usamos la misma lógica: si paramos, el error visual es 0
    if arrival_idx != -1 and i >= arrival_idx:
        e = 0.0
    else:
        eta = eta_function(p[0], p[1])
        e = eta - (T_controlador - t)
    errors.append(e)

# Gráficos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Panel 1
ax1.set_xlim(-6, 1)
ax1.set_ylim(-5, 5)
ax1.grid(True, alpha=0.3)
ax1.set_title(f"Trayectoria (Obj: {T_user}s)")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

circle = plt.Circle((0, 0), RADIO_META, color='r', fill=True, alpha=0.3)
ax1.add_patch(circle)

xx = np.linspace(-6, 1, 20)
yy = np.linspace(-5, 5, 20)
Xg, Yg = np.meshgrid(xx, yy)
Ug = Xg**2 - Yg**2
Vg = 2*Xg*Yg
Mg = np.sqrt(Ug**2 + Vg**2); Mg[Mg==0]=1
ax1.quiver(Xg, Yg, Ug/Mg, Vg/Mg, color='lightgray', scale=30, alpha=0.5)

trail, = ax1.plot([], [], 'b-', lw=1.5)
robot, = ax1.plot([], [], 'bo', ms=6)

time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, va='top', ha='left',
                     bbox=dict(boxstyle="round", facecolor='white', alpha=0.9, edgecolor='gray'))

# Panel 2
ax2.set_xlim(0, t_span[1])
ax2.grid(True, alpha=0.3)
ax2.set_title("Error de Tiempo (e)")
ax2.set_xlabel("Tiempo (s)")
ax2.axvline(T_user, c='g', ls='--', alpha=0.5)
ax2.axhline(0, c='k', ls=':', alpha=0.5)

err_line, = ax2.plot([], [], 'r-', lw=1.5)
err_dot, = ax2.plot([], [], 'ko', ms=4)

max_e = max(np.max(errors), 0.2)
min_e = min(np.min(errors), -0.2)
ax2.set_ylim(min_e*1.2, max_e*1.2)

def update(frame):
    # Lógica de congelación visual
    if arrival_idx != -1 and frame >= arrival_idx:
        display_t = real_arrival_time
        status = "LLEGADA (STOP)"
        col = 'green'
        idx_use = arrival_idx
        
        # FIX VISUAL DE LA GRÁFICA DE ERROR:
        # En el momento de la llegada, forzamos que se vea plano
        # para evitar el último punto 'colgando' si hubo micro-error.
        current_errors = errors[:frame+1]
        current_errors[-1] = 0.0 
    else:
        display_t = t_anim[frame]
        status = "EN CAMINO"
        col = 'gray'
        idx_use = frame
        if display_t > T_user + 0.02:
            status = "RETRASO"
            col = 'red'
        current_errors = errors[:frame+1]

    trail.set_data(X_traj[:idx_use+1], Y_traj[:idx_use+1])
    robot.set_data([X_traj[idx_use]], [Y_traj[idx_use]])
    
    err_line.set_data(t_anim[:frame+1], current_errors)
    err_dot.set_data([display_t], [current_errors[-1]])
    
    time_text.set_text(f"Obj: {T_user:.3f}s\nSim: {display_t:.3f}s\n{status}")
    time_text.get_bbox_patch().set_edgecolor(col)
    time_text.get_bbox_patch().set_linewidth(2)
    
    return trail, robot, err_line, err_dot, time_text

step = 2
frames = range(0, len(t_anim), step)
ani = FuncAnimation(fig, update, frames=frames, interval=30, blit=False)

print("Guardando GIF 'trayectoria_suave.gif'...")
try:
    ani.save("trayectoria_3.gif", writer=PillowWriter(fps=30))
    print("¡Hecho!")
except Exception as e:
    print(f"Error: {e}")

plt.show()