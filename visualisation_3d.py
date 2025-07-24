from manim import *
import numpy as np
from scipy.integrate import solve_ivp

class SchwarzschildOrbit(ThreeDScene):
    def compute_trajectory(self, M, L, r0, E, inward, e=1):
        def compute_pr(E, L, r, M, inward,e):
            f = 1 - 2*M/r
            Veff = f * (e + L**2 / r**2)
            rad_term = E**2 - Veff
            if rad_term < 0:
                raise ValueError("Invalid radial term; try adjusting E or r.")
            pr = (r**2 / L) * np.sqrt(rad_term)
            return -pr if inward else pr

        def geodesic(phi, y):
            r, pr = y
            f = 1 - 2*M/r
            dV_eff_dr = (2*M / r**2) * (e + L**2 / r**2) - f * (2 * L**2 / r**3)
            d2r_dphi2 = -0.5 * dV_eff_dr * (r**4 / L**2)
            return [pr, d2r_dphi2]

        pr0 = compute_pr(E, L, r0, M, inward, e)
        sol = solve_ivp(geodesic, (0, 12 * np.pi), [r0, pr0], max_step=0.05)
        phi_vals = sol.t
        r_vals = sol.y[0]
        x_vals = r_vals * np.cos(phi_vals)
        y_vals = r_vals * np.sin(phi_vals)
        return x_vals, y_vals

    def construct(self):
        self.camera.background_color = WHITE
        self.set_camera_orientation(phi=50 * DEGREES, theta=10 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.0)

        # Schwarzschild parameters
        M = 0.5  # Mass of black hole 
        L = 2.5867  # Angular momentum 
        r0 = 5 # Initial radius
        e = 1 # massive particle or photon
        E = 1.0595         # Input Energy 
        inward = False

        x_vals, y_vals = self.compute_trajectory(M, L, r0, E, inward, e)

        scale = 0.5
        x_vals = x_vals * scale
        y_vals = y_vals * scale
        
        # Tilt angle for the orbital plane (in degrees)
        tilt_angle = 30  # Change this to adjust the tilt
        
        # Convert to radians
        theta = tilt_angle * DEGREES
        
        # Rotate coordinates to tilt the plane
        z_vals = np.zeros_like(x_vals)
        # Apply rotation around x-axis
        y_vals_tilted = y_vals * np.cos(theta) - z_vals * np.sin(theta)
        z_vals_tilted = y_vals * np.sin(theta) + z_vals * np.cos(theta)

        axes = ThreeDAxes(x_range=[-10,10], y_range=[-10,10], z_range=[-5,5])
        self.add(axes)

        # Create grid lines (tilted)
        for angle in np.linspace(0, 2 * np.pi, 12):  
            # Create line in xy plane
            x_start = 3 * np.cos(angle)
            y_start = 3 * np.sin(angle)
            x_end = 10 * np.cos(angle)
            y_end = 10 * np.sin(angle)
            
            # Apply rotation to tilt the lines
            y_start_tilted = y_start * np.cos(theta)
            z_start_tilted = y_start * np.sin(theta)
            y_end_tilted = y_end * np.cos(theta)
            z_end_tilted = y_end * np.sin(theta)
            
            line = Line(
                start=[x_start, y_start_tilted, z_start_tilted],
                end=[x_end, y_end_tilted, z_end_tilted],
                stroke_opacity=0.8,
                stroke_width=1,
                color=BLACK
            )
            self.add(line)

        # Create circles (tilted)
        for r in np.linspace(3, 10, 8):  
            # Create circle in xy plane
            circ = Circle(radius=r * scale, color=BLACK, stroke_opacity=0.8, stroke_width=1.5)
            # Rotate the circle
            circ.rotate(theta, RIGHT, about_point=ORIGIN)
            self.add(circ)

        black_hole = Sphere(
           radius=2*M*scale,
           resolution=(32, 32),  
           fill_color=BLACK,  
           fill_opacity=1.0,
           stroke_width=0
        )
        self.add(black_hole)

        particle = Sphere(radius=0.05, resolution=(12, 12)).set_color(BLACK)
        particle.move_to([x_vals[0], y_vals_tilted[0], z_vals_tilted[0]])
        self.add(particle)

        path = VMobject(color=RED, stroke_width=3)
        path.start_new_path([x_vals[0], y_vals_tilted[0], z_vals_tilted[0]])
        self.add(path)

        def update_path(mob):
            idx = max(1, min(len(x_vals), int(self.time * 10)))
            points = np.column_stack([
                x_vals[:idx],
                y_vals_tilted[:idx],
                z_vals_tilted[:idx]
            ])
            mob.set_points_as_corners(points)
            if idx > 0:  
                particle.move_to(points[-1])

        path.add_updater(update_path)
        self.add(path)

        self.wait(12)  
        path.remove_updater(update_path)