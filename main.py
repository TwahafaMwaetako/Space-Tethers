import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from scipy.integrate import odeint

class SpaceTetherSimulation:
    def __init__(self):
        """
        Initialize the Space Tether Simulation.

        This class simulates the dynamics of a space tether in orbit around the Earth.
        """
        self.G = 6.67430e-11  # Gravitational constant
        self.M_earth = 5.97e24  # Mass of Earth (kg)
        self.R_earth = 6371e3  # Radius of Earth (m)
        self.tether_length = 100e3  # Length of tether (m)
        self.tether_mass = 1000  # Mass of tether (kg)
        self.payload_mass = 1000  # Mass of payload (kg)
        self.orbit_radius = self.R_earth + 300e3  # Initial orbit radius (m)
        self.orbit_period = 2 * np.pi * np.sqrt(self.orbit_radius**3 / (self.G * self.M_earth))
        self.initial_angular_velocity = 2 * np.pi / self.orbit_period

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title('Space Tether Simulation')
        plt.subplots_adjust(left=0.1, bottom=0.35)
        
        self.setup_plot()
        self.setup_sliders()
        self.setup_buttons()
        
        self.ani = None
        self.simulate()

    def setup_plot(self):
        """
        Set up the plot for the simulation.

        This function sets up the plot axes, draws the Earth, and initializes the tether plot.
        """
        self.ax.set_xlim(-self.orbit_radius*1.5, self.orbit_radius*1.5)
        self.ax.set_ylim(-self.orbit_radius*1.5, self.orbit_radius*1.5)
        self.ax.set_aspect('equal', 'box')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_facecolor('#E6E6E6')
        
        # Draw Earth
        earth_circle = plt.Circle((0, 0), self.R_earth, color='royalblue', alpha=0.8)
        self.ax.add_artist(earth_circle)
        
        # Initialize tether plot
        self.tether_line, = self.ax.plot([], [], 'ro-', lw=2, markersize=8)
        self.com_point, = self.ax.plot([], [], 'go', markersize=10)
        
        # Text annotations
        self.time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)

    def setup_sliders(self):
        """
        Set up the sliders for the simulation.

        This function sets up the sliders for the tether length, orbit height, and payload mass.
        """
        slider_color = 'lightgoldenrodyellow'
        ax_length = plt.axes([0.1, 0.2, 0.8, 0.03], facecolor=slider_color)
        ax_orbit = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor=slider_color)
        ax_mass = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=slider_color)
        
        self.length_slider = Slider(ax_length, 'Tether Length (km)', 10, 1000, valinit=self.tether_length/1000)
        self.orbit_slider = Slider(ax_orbit, 'Orbit Height (km)', 200, 2000, valinit=(self.orbit_radius - self.R_earth)/1000)
        self.mass_slider = Slider(ax_mass, 'Payload Mass (kg)', 100, 10000, valinit=self.payload_mass)
        
        self.length_slider.on_changed(self.update)
        self.orbit_slider.on_changed(self.update)
        self.mass_slider.on_changed(self.update)

    def setup_buttons(self):
        """
        Set up the buttons for the simulation.

        This function sets up the simulate button.
        """
        ax_simulate = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.button_simulate = Button(ax_simulate, 'Simulate', color='lightblue', hovercolor='0.975')
        self.button_simulate.on_clicked(self.simulate)

    def update(self, val):
        """
        Update the simulation parameters.

        This function updates the tether length, orbit height, and payload mass based on the slider values.
        """
        self.tether_length = self.length_slider.val * 1000
        self.orbit_radius = self.R_earth + self.orbit_slider.val * 1000
        self.payload_mass = self.mass_slider.val
        self.orbit_period = 2 * np.pi * np.sqrt(self.orbit_radius**3 / (self.G * self.M_earth))
        self.initial_angular_velocity = 2 * np.pi / self.orbit_period
        
        self.ax.set_xlim(-self.orbit_radius*1.5, self.orbit_radius*1.5)
        self.ax.set_ylim(-self.orbit_radius*1.5, self.orbit_radius*1.5)

    def tether_dynamics(self, state, t):
        """
        Calculate the dynamics of the tether.

        This function calculates the gravitational and centrifugal forces on the tether and updates the state of the system.

        Parameters:
        state (list): The current state of the system (x, y, vx, vy, theta, omega)
        t (float): The current time

        Returns:
        list: The updated state of the system
        """
        x, y, vx, vy, theta, omega = state
        
        r = np.sqrt(x**2 + y**2)
        
        # Gravitational force
        Fg_x = -self.G * self.M_earth * (self.tether_mass + self.payload_mass) * x / r**3
        Fg_y = -self.G * self.M_earth * (self.tether_mass + self.payload_mass) * y / r**3
        
        # Centrifugal force
        Fc_x = (self.tether_mass + self.payload_mass) * omega**2 * x
        Fc_y = (self.tether_mass + self.payload_mass) * omega**2 * y
        
        # Equations of motion
        dx_dt = vx
        dy_dt = vy
        dvx_dt = (Fg_x + Fc_x) / (self.tether_mass + self.payload_mass)
        dvy_dt = (Fg_y + Fc_y) / (self.tether_mass + self.payload_mass)
        dtheta_dt = omega
        domega_dt = -2 * (vx * y - vy * x) * omega / r**2
        
        return [dx_dt, dy_dt, dvx_dt, dvy_dt, dtheta_dt, domega_dt]

    def simulate(self, event=None):
        """
        Run the simulation.

        This function runs the simulation by solving the equations of motion using odeint and updating the plot.

        Parameters:
        event (None): Not used
        """
        if self.ani is not None:
            self.ani.event_source.stop()
        
        initial_state = [self.orbit_radius, 0, 0, self.orbit_radius * self.initial_angular_velocity, 0, self.initial_angular_velocity]
        self.t = np.linspace(0, self.orbit_period * 2, 1000)
        self.solution = odeint(self.tether_dynamics, initial_state, self.t)
        
        self.ani = FuncAnimation(self.fig, self.animate, frames=len(self.t),
                                 interval=50, blit=True)
        plt.draw()

    def animate(self, i):
        """
        Update the plot for the current frame.

        This function updates the plot by drawing the tether and payload at the current position.

        Parameters:
        i (int): The current frame number

        Returns:
        list: The updated plot elements
        """
        x, y = self.solution[i, 0], self.solution[i, 1]
        theta = self.solution[i, 4]
        
        tether_x = x + self.tether_length * np.sin(theta)
        tether_y = y - self.tether_length * np.cos(theta)
        
        self.tether_line.set_data([x, tether_x], [y, tether_y])
        self.com_point.set_data(x, y)
        self.time_text.set_text(f'Time: {self.t[i]:.2f} s')
        
        return self.tether_line, self.com_point, self.time_text

    def run(self):
        """
        Run the simulation.

        This function runs the simulation by calling the simulate function.
        """
        plt.show()

# Run the simulation
simulation = SpaceTetherSimulation()
simulation.run()