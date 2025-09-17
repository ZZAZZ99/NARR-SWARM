import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# --- Parámetros de la malla y rango ---
x = np.linspace(-3, 3, 200)
y = np.linspace(0, 6, 200)
X, Y = np.meshgrid(x, y)

# Campo no normalizado (para la dinámica)
Fx = X**2 - Y**2
Fy = 2 * X * Y

# Para representar direcciones (igual que en el artículo) normalizamos solo para la visualización
magnitude = np.sqrt(Fx**2 + Fy**2)
Fx_dir = np.divide(Fx, magnitude, where=magnitude!=0)
Fy_dir = np.divide(Fy, magnitude, where=magnitude!=0)

# --- Sistema dinámico (EDO) ---
def campo(t, z):
    x, y = z
    return [x**2 - y**2, 2*x*y]

# Condición inicial y solución (dense_output para evaluar en tiempos arbitrarios)
z0 = [1.0, 2.0]         # condición inicial (puedes cambiarla)
t_span = (0.0, 4.0)     # intervalo de tiempo de integración
sol = solve_ivp(campo, t_span, z0, max_step=0.005, dense_output=True)

# Crear tiempos para la animación (más finos que la malla temporal de solve_ivp)
t_vals = np.linspace(t_span[0], t_span[1], 800)
x_vals, y_vals = sol.sol(t_vals)

# --- Preparar figura ---
fig, ax = plt.subplots(figsize=(7,7))
ax.streamplot(X, Y, Fx_dir, Fy_dir, density=1.2, arrowsize=1)
ax.set_xlim(-3, 3)
ax.set_ylim(0, 6)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Animación: partícula en F(x,y) = (x² - y², 2xy) — trayectoria en rojo")

# Línea que irá acumulando la trayectoria y punto que se mueve
traj_line, = ax.plot([], [], 'r-', linewidth=1.5)   # trayectoria completa hasta el instante
particle_dot, = ax.plot([], [], 'ro', markersize=6) # partícula

# (opcional) marcar posición inicial
ax.plot(z0[0], z0[1], 'go', label='Inicial')
ax.legend()

# --- Funciones para animación ---
def init():
    traj_line.set_data([], [])
    particle_dot.set_data([], [])
    return traj_line, particle_dot

def update(i):
    # hasta el frame i dibujamos la parte de la trayectoria desde el inicio hasta ese tiempo
    traj_line.set_data(x_vals[:i+1], y_vals[:i+1])
    particle_dot.set_data(x_vals[i], y_vals[i])
    return traj_line, particle_dot

ani = FuncAnimation(fig, update, frames=len(t_vals),
                    init_func=init, blit=True, interval=20)

plt.show()
