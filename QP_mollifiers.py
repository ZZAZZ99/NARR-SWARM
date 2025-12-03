import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from qpsolvers import solve_qp

# --- 1. DEFINICIÓN DEL MOLLIFIER (Paper Example 1) ---
def standard_mollifier(x):
    """
    Función phi(x) definida en Example 1 del paper.
    Soporte compacto en [-1, 1], C^infinito.
    """
    if abs(x) >= 1.0:
        return 0.0
    return np.exp(-1.0 / (1.0 - x**2))

# Constante de normalización c1 para que la integral sea 1
NORMALIZATION_C, _ = quad(standard_mollifier, -1, 1)

def phi_epsilon(t, epsilon):
    """
    Mollifier escalado phi_epsilon(x) = (1/eps) * phi(x/eps).
    """
    return (1.0 / epsilon) * (standard_mollifier(t / epsilon) / NORMALIZATION_C)

# --- 2. CÁLCULO VECTORIZADO (RÁPIDO) ---
def mollify_trajectory(x_pts, y_pts, epsilon, num_samples=500, return_indices=False):
    """
    Realiza la convolución numérica F = f * phi_epsilon usando NumPy.
    Versión optimizada vectorizada para evitar bucles lentos.
    """
    # Parametrización por longitud de arco
    points = np.vstack((x_pts, y_pts)).T
    dists = np.sqrt(np.diff(x_pts)**2 + np.diff(y_pts)**2)
    times = np.zeros(len(x_pts))
    times[1:] = np.cumsum(dists)
    total_len = times[-1]
    
    # Malla de evaluación (t) y de integración (s)
    t_fine = np.linspace(0, total_len, num_samples)
    
    # Si necesitamos saber dónde caen los waypoints originales en el tiempo t_fine
    waypoint_indices = []
    if return_indices:
        for t_wp in times:
            # Buscar el índice en t_fine más cercano al tiempo del waypoint
            idx = (np.abs(t_fine - t_wp)).argmin()
            waypoint_indices.append(idx)

    integration_steps = 50 
    s_vals = np.linspace(-epsilon, epsilon, integration_steps)
    ds = s_vals[1] - s_vals[0]
    
    # Pesos del kernel (simétrico)
    weights = np.array([phi_epsilon(s, epsilon) for s in s_vals])
    
    # Vectorización: T_grid [num_samples, integration_steps]
    # Matriz gigante de tiempos a consultar: T_eval = t - s
    t_grid = t_fine[:, np.newaxis] - s_vals[np.newaxis, :]
    t_grid = t_grid % total_len # Manejo de periodicidad
    
    # Interpolación Lineal Vectorizada
    t_flat = t_grid.flatten()
    # Búsqueda rápida de índices
    indices = np.searchsorted(times, t_flat) - 1
    indices = np.clip(indices, 0, len(points) - 2)
    
    p_start = points[indices]
    p_end = points[indices + 1]
    t_start = times[indices]
    t_end = times[indices + 1]
    
    denom = t_end - t_start
    denom[denom < 1e-9] = 1.0 
    
    ratios = (t_flat - t_start) / denom
    p_interp_flat = p_start + (p_end - p_start) * ratios[:, np.newaxis]
    p_interp = p_interp_flat.reshape(num_samples, integration_steps, 2)
    
    # Convolución (Suma ponderada por los pesos del mollifier)
    x_mollified = np.sum(p_interp[:, :, 0] * weights, axis=1) * ds
    y_mollified = np.sum(p_interp[:, :, 1] * weights, axis=1) * ds
    
    if return_indices:
        return x_mollified, y_mollified, t_fine, waypoint_indices
    return x_mollified, y_mollified, t_fine

# --- 3. ALGORITMO DE INTERPOLACIÓN FORZADA ---
def force_interpolation(x_target, y_target, epsilon, max_iter=15):
    """
    Ajusta iterativamente los puntos de entrada (Control Virtual) para que 
    la curva suavizada resultante pase EXACTAMENTE por los puntos objetivo.
    """
    print(f"   -> Ajustando puntos de control para interpolación (eps={epsilon})...")
    
    # Puntos virtuales iniciales = Puntos objetivo
    # IMPORTANTE: Asegurar que sean float para permitir decimales
    x_virtual = x_target.copy().astype(float)
    y_virtual = y_target.copy().astype(float)
    
    # Factor de ganancia (1.1 ayuda a converger más rápido)
    gain = 1.1 
    
    for i in range(max_iter):
        # 1. Calcular curva actual con los puntos virtuales actuales
        mx, my, _, wp_idx = mollify_trajectory(x_virtual, y_virtual, epsilon, num_samples=1000, return_indices=True)
        
        # 2. Obtener las posiciones resultantes en los tiempos de los waypoints
        # Nota: wp_idx[:-1] porque el último punto es el cierre (repetido)
        curr_x_at_wp = mx[wp_idx]
        curr_y_at_wp = my[wp_idx]
        
        # 3. Calcular error (Target - Actual)
        err_x = x_target - curr_x_at_wp
        err_y = y_target - curr_y_at_wp
        
        max_err = np.max(np.sqrt(err_x**2 + err_y**2))
        
        # 4. Corregir los puntos virtuales (Inverse Mollification)
        x_virtual += err_x * gain
        y_virtual += err_y * gain
        
        if max_err < 1e-3: # Tolerancia de 1 milímetro
            print(f"      Convergencia alcanzada en {i+1} iteraciones. Error max: {max_err:.4f}m")
            break
    
    if max_err >= 1e-3:
        print(f"      [Aviso] Max iteraciones alcanzadas. Error residual: {max_err:.4f}m")
            
    return x_virtual, y_virtual

def draw_error_bar(ax_plot, xi, yi, dxe, dye):
    """ Dibuja la 'puerta' o intervalo de incertidumbre """
    ax_plot.plot([xi-dxe, xi+dxe], [yi-dye, yi+dye], 'k-', lw=1.5, alpha=0.6)
    # Pequeños topes visuales
    norm = math.hypot(dxe, dye)
    if norm > 0:
        cap_x = -dye/norm * 0.05
        cap_y = dxe/norm * 0.05
        ax_plot.plot([xi-dxe-cap_x, xi-dxe+cap_x], [yi-dye-cap_y, yi-dye+cap_y], 'k-', lw=1)
        ax_plot.plot([xi+dxe-cap_x, xi+dxe+cap_x], [yi+dye-cap_y, yi+dye+cap_y], 'k-', lw=1)

# --- 4. ENTRADA DE DATOS Y CONFIGURACIÓN ---
def solicitar_datos():
    n = int(input("Número de puntos (sin cerrar curva): "))
    x_c, y_c, s_dir, t_dir = [], [], [], []
    for i in range(n):
        x_c.append(float(input(f"Pt {i+1} x: ")))
        y_c.append(float(input(f"Pt {i+1} y: ")))
        s_dir.append(float(input(f"Pt {i+1} sigma: ")))
        t_dir.append(float(input(f"Pt {i+1} theta: ")))
    return (np.array(x_c), np.array(y_c), np.array(s_dir), np.array(t_dir))

print("--- CONFIGURACIÓN ---")
uso_default = input("¿Usar datos por defecto? (s/n): ")
if uso_default.lower() in ('s','si','y','yes'):
    # Datos por defecto
    x_central = np.array([0, 1, 2, 1, 0])
    y_central = np.array([0, 1, 2, 3, 2])
    sigma_dir = np.array([0.15, 0.10, 0.20, 0.15, 0.10])
    theta_dir = np.array([np.pi/6, np.pi/4, np.pi/3, np.pi/8, np.pi/10])
else:
    x_central, y_central, sigma_dir, theta_dir = solicitar_datos()

# --- CORRECCIÓN CRÍTICA: ASEGURAR TIPO FLOAT ---
# Esto evita el error "Cannot cast ufunc 'add' output from dtype('float64') to dtype('int32')"
x_central = x_central.astype(float)
y_central = y_central.astype(float)

epsilon_val = float(input("Valor de Epsilon (suavizado, ej. 0.5): ") or 0.5)

# Cerrar curvas (Target) - Los puntos "reales" por donde queremos pasar
cerrar = lambda arr: np.append(arr, arr[0])
x_target = cerrar(x_central)
y_target = cerrar(y_central)
dx_err = sigma_dir * np.cos(theta_dir)
dy_err = sigma_dir * np.sin(theta_dir)

# Calcular fronteras objetivos (targets)
x_plus_target = cerrar(x_central + dx_err)
y_plus_target = cerrar(y_central + dy_err)
x_minus_target = cerrar(x_central - dx_err)
y_minus_target = cerrar(y_central - dy_err)

print("\n--- 1. CÁLCULO DE CURVAS CON INTERPOLACIÓN FORZADA ---")
# Calculamos los puntos virtuales (desplazados) para lograr la interpolación
xv_c, yv_c = force_interpolation(x_target, y_target, epsilon_val)
xv_p, yv_p = force_interpolation(x_plus_target, y_plus_target, epsilon_val)
xv_m, yv_m = force_interpolation(x_minus_target, y_minus_target, epsilon_val)

# Generamos las curvas finales usando los puntos virtuales
print("   -> Generando geometría fina final...")
mx_c, my_c, _ = mollify_trajectory(xv_c, yv_c, epsilon_val)
mx_p, my_p, _ = mollify_trajectory(xv_p, yv_p, epsilon_val)
mx_m, my_m, _ = mollify_trajectory(xv_m, yv_m, epsilon_val)

# --- VISUALIZACIÓN 1: FRONTERAS E INTERPOLACIÓN ---
plt.figure("Figura 1: Fronteras e Interpolación", figsize=(8, 6))
plt.plot(mx_p, my_p, 'g-', lw=2, label='Frontera +')
plt.plot(mx_m, my_m, 'r-', lw=2, label='Frontera -')
plt.plot(mx_c, my_c, '--', color='blue', alpha=0.5, label='Central')

# Dibujar puntos reales vs virtuales para visualizar el "truco"
plt.scatter(x_central, y_central, c='black', s=50, zorder=10, label='Waypoints (Objetivo)')
plt.scatter(xv_c[:-1], yv_c[:-1], c='gray', s=20, marker='x', label='Control Virtual (Desplazados)')

for xi, yi, dxe, dye in zip(x_central, y_central, dx_err, dy_err):
    draw_error_bar(plt.gca(), xi, yi, dxe, dye)

plt.title(f"Mollifiers con Interpolación Forzada (Eps={epsilon_val})\nLa curva pasa exactamente por los puntos negros")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.axis('equal')

# --- 2. OPTIMIZACIÓN QP (Minimizar Curvatura Global) ---
print("\n--- 2. OPTIMIZACIÓN DE CURVATURA ---")
# Usamos las curvas mollificadas como base para la optimización
p1 = np.vstack((mx_p, my_p)).T
p2 = np.vstack((mx_m, my_m)).T
n_q = len(p1)

# Matriz de diferencias finitas para segunda derivada (curvatura)
D = np.zeros((n_q-2, n_q))
for i in range(n_q-2):
    D[i, i] = 1; D[i, i+1] = -2; D[i, i+2] = 1

dd = []
for P in (p1, p2):
    dd_x = D @ P[:,0]
    dd_y = D @ P[:,1]
    dd.append((dd_x, dd_y))

# Construcción de matriz Hessiana H = Integral(k^2)
H = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        xi, yi = dd[i]; xj, yj = dd[j]
        H[i, j] = xi @ xj + yi @ yj

try:
    # Configuración Solver QP
    P_qp = 2 * H
    q_qp = np.zeros(2)
    A_eq = np.ones((1, 2)); b_eq = np.array([1.0]) # Suma = 1
    lb = np.zeros(2); ub = np.ones(2)              # Límites 0..1
    
    lambda_opt = solve_qp(P_qp, q_qp, None, None, A_eq, b_eq, lb, ub, solver='highs')
    
    if lambda_opt is None: 
        print("   [Aviso] Solver retornó None. Usando promedio.")
        lambda_opt = np.array([0.5, 0.5])
    else:
        print(f"   -> Pesos óptimos encontrados: λ+ = {lambda_opt[0]:.3f}, λ- = {lambda_opt[1]:.3f}")
except Exception as e:
    print(f"   -> Error en Solver ({e}). Usando promedio.")
    lambda_opt = np.array([0.5, 0.5])

# Generar curva final
p_fit = lambda_opt[0]*p1 + lambda_opt[1]*p2

# --- VISUALIZACIÓN 2: RESULTADO FINAL ---
plt.figure("Figura 2: Trayectoria Óptima", figsize=(8, 6))

plt.plot(mx_p, my_p, 'g--', alpha=0.2)
plt.plot(mx_m, my_m, 'r--', alpha=0.2)
for xi, yi, dxe, dye in zip(x_central, y_central, dx_err, dy_err):
    draw_error_bar(plt.gca(), xi, yi, dxe, dye)

plt.plot(p_fit[:,0], p_fit[:,1], color='purple', lw=3, label='Trayectoria Óptima')
plt.scatter(x_central, y_central, c='red', s=40, zorder=5, label='Waypoints')

plt.title(f'Trayectoria Final (Min Curvatura)\nPesos: λ1={lambda_opt[0]:.2f}, λ2={lambda_opt[1]:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.show()