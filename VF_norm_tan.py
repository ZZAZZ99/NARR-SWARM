import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# ------------------------------
# Solicitar entrada al usuario
# ------------------------------
r = float(input("Introduce la distancia radial r: "))
theta_mult_pi = float(input("Introduce el ángulo theta como múltiplo de pi: "))
theta = theta_mult_pi * np.pi
x0 = r * np.cos(theta)
y0 = r * np.sin(theta)

# Ahora se introducen los cuadrados
v_tan_sq = float(input("Introduce v_tan^2: "))
v_norm_sq = float(input("Introduce v_norm^2: "))

# Verificar que sumen 1 (con tolerancia numérica)
if abs(v_tan_sq + v_norm_sq - 1.0) > 1e-6:
    raise ValueError("Error: debe cumplirse que v_tan^2 + v_norm^2 = 1")

# Convertir a valores (se toma la raíz positiva)
v_tan = np.sqrt(v_tan_sq)
v_norm = np.sqrt(v_norm_sq)

# ------------------------------
# Crear malla
# ------------------------------
x = np.linspace(-3, 3, 200)
y = np.linspace(0, 6, 200)
X, Y = np.meshgrid(x, y)

Fx = X**2 - Y**2
Fy = 2*X*Y
magnitude = np.sqrt(Fx**2 + Fy**2)
Fx_dir = np.divide(Fx, magnitude, where=magnitude!=0)
Fy_dir = np.divide(Fy, magnitude, where=magnitude!=0)

# ------------------------------
# Función tangencial y normal
# ------------------------------
def tangencial_normal(x, y):
    Fx_val = x**2 - y**2
    Fy_val = 2*x*y
    mag = np.sqrt(Fx_val**2 + Fy_val**2)
    if mag == 0:
        t_hat = np.array([1.0, 0.0])
        n_hat = np.array([0.0, 1.0])
    else:
        t_hat = np.array([Fx_val, Fy_val]) / mag
        n_hat = np.array([-Fy_val, Fx_val]) / mag
    return t_hat, n_hat

# ------------------------------
# ODE con velocidad variable
# ------------------------------
def campo_velocidad_variable(t, z):
    x, y = z
    t_hat, n_hat = tangencial_normal(x, y)
    return v_tan * t_hat + v_norm * n_hat

# ------------------------------
# Integración
# ------------------------------
t_span = (0.0, 10.0)
sol = solve_ivp(campo_velocidad_variable, t_span, [x0, y0],
                max_step=0.005, dense_output=True)
t_vals = np.linspace(t_span[0], t_span[1], 2000)
x_vals, y_vals = sol.sol(t_vals)

dist_to_origin = np.sqrt(x_vals**2 + y_vals**2)
threshold = 0.05
arrival_indices = np.where(dist_to_origin <= threshold)[0]
if len(arrival_indices) > 0:
    arrival_index = arrival_indices[0]
    arrival_time = t_vals[arrival_index]
else:
    arrival_index = len(t_vals) - 1
    arrival_time = None

# ------------------------------
# Preparar figura y animación
# ------------------------------
fig, ax = plt.subplots(figsize=(7,7))
ax.streamplot(X, Y, Fx_dir, Fy_dir, density=1.2, arrowsize=1)
ax.set_xlim(-3, 3)
ax.set_ylim(0, 6)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Trayectoria con velocidad tangencial y normal variable")

traj_line, = ax.plot([], [], 'r-', linewidth=1.5)
particle_dot, = ax.plot([], [], 'ro', markersize=6)

# Puntos inicial y origen
ax.scatter([x0], [y0], color='green', edgecolor='black', s=100, zorder=5, label='Inicial')
ax.scatter([0], [0], color='blue', edgecolor='black', s=100, zorder=5, label='Origen')

# Leyenda en esquina superior derecha
leg = ax.legend(loc='upper right')

# Copiar estilo del borde de la leyenda
leg_edge = leg.get_frame()
legend_bbox_style = dict(facecolor='white',
                         edgecolor=leg_edge.get_edgecolor(),
                         linewidth=leg_edge.get_linewidth(),
                         alpha=0.8)

# Texto ETA en esquina superior izquierda con mismo borde
eta_text = ax.text(0.03, 0.95, '', transform=ax.transAxes,
                   ha='left', va='top',
                   bbox=legend_bbox_style)

# ------------------------------
# Funciones para animación
# ------------------------------
def init():
    traj_line.set_data([], [])
    particle_dot.set_data([], [])
    eta_text.set_text('')
    return traj_line, particle_dot, eta_text

def update(i):
    traj_line.set_data(x_vals[:i+1], y_vals[:i+1])
    particle_dot.set_data(x_vals[i], y_vals[i])
    if arrival_time is not None and t_vals[i] >= arrival_time:
        eta_text.set_text(f"ETA: {arrival_time:.2f} s")
    else:
        eta_text.set_text(f"ETA: {t_vals[i]:.2f} s")
    return traj_line, particle_dot, eta_text

ani = FuncAnimation(fig, update, frames=arrival_index+1,
                    init_func=init, blit=True, interval=20)

# ------------------------------
# Guardar animación según usuario
# ------------------------------
guardar = input("¿Quieres guardar la animación? (s/n): ").strip().lower()
if guardar == "s":
    formato = input("Elige formato (gif/mp4): ").strip().lower()
    if formato == "gif":
        ani.save("trayectoria.gif", writer="pillow", fps=30)
        print("Animación guardada como trayectoria.gif")
    elif formato == "mp4":
        ani.save("trayectoria.mp4", writer="ffmpeg", fps=30)
        print("Animación guardada como trayectoria.mp4")
    else:
        print("Formato no válido. No se guardó la animación.")

plt.show()
