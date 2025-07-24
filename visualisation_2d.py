import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

def main():
    # Schwarzschild parameters
    M = 0.5  # Mass of black hole 
    L = 1.8868  # Angular momentum 
    r0 = 5 # Initial radius
    e = 1 # massive particle or photon
    E = 0.9560         # Input Energy 
    inward = True



    def compute_pr(E, L, r, M=1, inward=True,e=1):
        f = 1 - 2*M/r
        Veff = f * (e + L**2 / r**2)
        rad_term = E**2 - Veff
        if rad_term < 0:
            raise ValueError(
            f"Initial radius r = {r} is not allowed for E = {E}, L = {L}. "
            f"Try increasing E or r, or decreasing L.\n"
            f"Computed V_eff = {Veff:.6f}, so sqrt(E^2 - Veff) = sqrt({E**2:.6f} - {Veff:.6f}) = imaginary"
        )
        pr = (r**2 / L) * np.sqrt(rad_term)
        return -pr if inward else pr



    def geodesic(phi, y):
        r, pr = y
        f = 1 - 2*M/r
        dV_eff_dr = (2*M / r**2) * (e + L**2 / r**2) - f * (2 * L**2 / r**3)
        d2r_dphi2 = -0.5 * dV_eff_dr * (r**4 / L**2)
        return [pr, d2r_dphi2]

    pr0 = compute_pr(E, L, r0, M, inward=inward,e=1)



    sol = solve_ivp(geodesic, (0, 20*np.pi), [r0, pr0], max_step=0.05)
    phi_vals = sol.t
    r_vals = sol.y[0]
    x_vals = r_vals * np.cos(phi_vals)
    y_vals = r_vals * np.sin(phi_vals)
    
    fig, ax = plt.subplots(figsize=(30, 20))
    ax.set_facecolor('white')
    ax.set_aspect('equal')
    ax.set_xlim(-r0-5, r0+5)
    ax.set_ylim(-r0-5, r0+5)
    ax.set_xticks([])
    ax.set_yticks([])
    

    ax.add_patch(plt.Circle((0, 0), 2*M, color='black', zorder=3))
    ax.add_patch(plt.Circle((0, 0), 2*M, color='white', 
                          fill=False))
    ax.add_patch(plt.Circle((0, 0), 3*M, color='yellow', 
                          fill=False, linestyle=':', alpha=0.5))
    ax.add_patch(plt.Circle((0, 0), 0.05, color='darkred', zorder=4))
    
    num_radials = 30
    num_circles = 12
    
    for angle in np.linspace(0, 2*np.pi, num_radials, endpoint=False):
        rs = np.linspace(2.1, r0+5, 100)
        x = rs * np.cos(angle)
        y = rs * np.sin(angle)
        ax.plot(x, y, color='gray', alpha=0.3, linewidth=0.5)
    
    for r in np.linspace(3, r0+5, num_circles):
        theta = np.linspace(0, 2*np.pi, 100)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.plot(x, y, color='gray', alpha=0.3, linewidth=0.5)
    
    orbit_path, = ax.plot([], [], color='red', lw=2)
    particle, = ax.plot([], [], 'o', color='white', markersize=8)
    
    def init():
        orbit_path.set_data([], [])
        particle.set_data([], [])
        return orbit_path, particle
    def animate(i):
        if i >= len(x_vals):
          return orbit_path, particle  
        orbit_path.set_data(x_vals[:i], y_vals[:i])
        particle.set_data([x_vals[i]], [y_vals[i]])
        return orbit_path, particle

    
   

    global ani
    ani = FuncAnimation(fig, animate, frames=len(phi_vals),
                       init_func=init, interval=4, blit=False)

    
    plt.title("Particle Orbit in Schwarzschild Spacetime", color='white', pad=20)
    plt.tight_layout()
    plt.show()
    return ani 


if __name__ == "__main__":
    ani = None  
    ani = main()
