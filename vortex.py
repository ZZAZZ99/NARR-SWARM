import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# --- Parámetros del Vórtice ---
c = 5.0       # Velocidad de succión radial (m/s)
omega = 3.0   # Velocidad angular de rotación (rad/s)

# --- Parámetros de la malla y rango ---
limit = 4
x = np.linspace(-limit, limit, 200)
y = np.linspace(-limit, limit, 200)
X, Y = np.meshgrid(x, y)

# Cálculo del radio para la malla (evitando división por cero)
R = np.sqrt(X**2 + Y**2)
R[R == 0] = 1e-9  # Pequeño epsilon para evitar errores en el origen

# Campo del Vórtice en Cartesianas:
# F(p) = -c * (p / ||p||) + omega * E * p
# Componente X: -c * (x/r) - omega * y
# Componente Y: -c * (y/r) + omega * x
Fx = -c * (X / R) - omega * Y
Fy = -c * (Y / R) + omega * X

# Normalizamos solo para la visualización de las flechas (streamplot)
magnitude = np.sqrt(Fx**2 + Fy**2)
Fx_dir = np.divide(Fx, magnitude, where=magnitude!=0)
Fy_dir = np.divide(Fy, magnitude, where=magnitude!=0)

# --- Sistema dinámico (EDO) ---
def campo_vortice(t, z):
    x, y = z
    r = np.sqrt(x**2 + y**2)
    
    # Evitar singularidad matemática exacta en el 0
    if r < 1e-3:
        return [0, 0]
        
    dx = -c * (x / r) - omega * y
    dy = -c * (y / r) + omega * x
    return [dx, dy]

# Condición inicial y solución
z0 = [3.0, 3.0]         # Empezamos lejos para ver la espiral
# El tiempo teórico para llegar al origen es T = r0 / c. 
# r0 = sqrt(3^2+3^2) ≈ 4.24. Con c=1, necesitamos al menos 4.5s.
t_span = (0.0, 5.0)     
sol = solve_ivp(campo_vortice, t_span, z0, max_step=0.01, dense_output=True)

# Crear tiempos para la animación
t_vals = np.linspace(t_span[0], t_span[1], 800)
x_vals, y_vals = sol.sol(t_vals)

# --- Preparar figura ---
fig, ax = plt.subplots(figsize=(7,7))

# HE ELIMINADO 'alpha=0.6' DE AQUÍ:
ax.streamplot(X, Y, Fx_dir, Fy_dir, density=1.2, arrowsize=1, color='gray') 

ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_xlabel("x")
ax.set_ylabel("y")
title_str = f"Vórtice: c={c} (radial), $\omega$={omega} (angular)"
ax.set_title(title_str)

# Línea que irá acumulando la trayectoria y punto que se mueve
traj_line, = ax.plot([], [], 'r-', linewidth=2, label='Trayectoria')
particle_dot, = ax.plot([], [], 'ro', markersize=8) 

# Marcar posición inicial y destino (origen)
ax.plot(z0[0], z0[1], 'go', label='Inicio')
ax.plot(0, 0, 'bx', markeredgewidth=2, label='Destino')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# --- Funciones para animación ---
def init():
    traj_line.set_data([], [])
    particle_dot.set_data([], [])
    return traj_line, particle_dot

def update(i):
    # Dibujar hasta el frame actual
    traj_line.set_data(x_vals[:i+1], y_vals[:i+1])
    particle_dot.set_data(x_vals[i], y_vals[i])
    return traj_line, particle_dot

ani = FuncAnimation(fig, update, frames=len(t_vals),
                    init_func=init, blit=True, interval=20)

plt.show()