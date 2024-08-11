from math import pi, exp
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.constants as const
import MDAnalysis as mda
import ipywidgets as widgets
from IPython.display import display
import tempfile
import webbrowser
import nglview as nv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# Material properties database
material_properties = {
    # Metals
    "Gold": {"density": 19.32, "refractive_index": 0.47, "melting_point": 1064, "heat_of_fusion": 12.55, "surface_tension": 0.5, "category": "Metal"},
    "Silver": {"density": 10.49, "refractive_index": 0.15, "melting_point": 961.8, "heat_of_fusion": 11.23, "surface_tension": 0.7, "category": "Metal"},
    "Platinum": {"density": 21.45, "refractive_index": 2.33, "melting_point": 1768, "heat_of_fusion": 19.7, "surface_tension": 1.8, "category": "Metal"},
    "Copper": {"density": 8.96, "refractive_index": 0.22, "melting_point": 1085, "heat_of_fusion": 13.00, "surface_tension": 0.6, "category": "Metal"},
    "Iron": {"density": 7.87, "refractive_index": 0.20, "melting_point": 1538, "heat_of_fusion": 13.81, "surface_tension": 1.872, "category": "Metal"},
    # Metal Oxides
    "Titanium Dioxide": {"density": 4.23, "refractive_index": 2.61, "melting_point": 1843, "heat_of_fusion": 66.9, "surface_tension": 0.0, "category": "Metal Oxide"},
    "Zinc Oxide": {"density": 5.61, "refractive_index": 2.00, "melting_point": 1975, "heat_of_fusion": 43.0, "surface_tension": 0.0, "category": "Metal Oxide"},
    
    # Semiconductors
    "Silicon": {"density": 2.33, "refractive_index": 3.49, "melting_point": 1414, "heat_of_fusion": 50.5, "surface_tension": 0.865, "category": "Semiconductor"},
    "Germanium": {"density": 5.32, "refractive_index": 4.00, "melting_point": 938.3, "heat_of_fusion": 36.9, "surface_tension": 0.785, "category": "Semiconductor"},
    
    # Carbon-based
    "Carbon Nanotube": {"density": 1.3, "refractive_index": 1.1, "melting_point": 3500, "heat_of_fusion": 0.0, "surface_tension": 0.0, "category": "Carbon-based"},
    "Graphene": {"density": 2.267, "refractive_index": 2.6, "melting_point": 4800, "heat_of_fusion": 0.0, "surface_tension": 0.0, "category": "Carbon-based"},
    
    # Polymers
    "Polystyrene": {"density": 1.05, "refractive_index": 1.59, "melting_point": 240, "heat_of_fusion": 0.0, "surface_tension": 0.033, "category": "Polymer"},
    "PLGA": {"density": 1.25, "refractive_index": 1.46, "melting_point": 185, "heat_of_fusion": 0.0, "surface_tension": 0.036, "category": "Polymer"},
    
    # Quantum Dots
    "CdSe/ZnS": {"density": 5.82, "refractive_index": 2.5, "melting_point": 1600, "heat_of_fusion": 0.0, "surface_tension": 0.0, "category": "Quantum Dot"},
    
    # Magnetic Materials
    "Iron Oxide (Fe3O4)": {"density": 5.17, "refractive_index": 2.42, "melting_point": 1538, "heat_of_fusion": 0.0, "surface_tension": 0.0, "category": "Magnetic Material"},
    
    # Ceramic
    "Hydroxyapatite": {"density": 3.16, "refractive_index": 1.64, "melting_point": 1670, "heat_of_fusion": 0.0, "surface_tension": 0.0, "category": "Ceramic"},
}

def enable_entries(*entries):
    for entry in entries:
        entry.config(state="normal")

def disable_entries(*entries):
    for entry in entries:
        entry.config(state="disabled")

def shape_changed(*args):
    shape = shape_var.get()
    if shape == "Cube":
        enable_entries(side_entry)
        disable_entries(radius_entry, height_entry, length_entry, width_entry)
    elif shape == "Sphere":
        enable_entries(radius_entry)
        disable_entries(side_entry, height_entry, length_entry, width_entry)
    elif shape == "Cylinder":
        enable_entries(radius_entry, height_entry)
        disable_entries(side_entry, length_entry, width_entry)
    elif shape == "Cuboid" or shape == "Rectangular Prism":
        enable_entries(length_entry, width_entry, height_entry)
        disable_entries(side_entry, radius_entry)

def calculate():
    shape = shape_var.get()
    material = material_var.get()
    density = material_properties[material]["density"]  # in g/cm^3
    heat_of_fusion = material_properties[material]["heat_of_fusion"] * 1e3  # J/mol
    surface_tension = material_properties[material]["surface_tension"]  # N/m
    melting_point_bulk = material_properties[material]["melting_point"]  # °C
      # Calculate additional properties
    zeta_potential = calculate_zeta_potential(radius, charge, temperature)
    van_der_waals_force = calculate_van_der_waals_force(radius, radius, hamaker_constant, distance)
    debye_length = calculate_debye_length(ionic_strength, temperature)
    
    # Update the material_properties_label with new properties
    material_properties_label.config(
        text=f"{existing_text}\n"
             f"Zeta Potential: {zeta_potential:.2f} mV\n"
             f"Van der Waals Force: {van_der_waals_force:.2e} N\n"
             f"Debye Length: {debye_length:.2e} m"
    )

volume, surface_area, mass = 0, 0, 0
try:
    if shape == "Cube":
        side = float(side_entry.get()) * 1e-9  # nm to meters
        volume = side ** 3  # m^3
        surface_area = 6 * side ** 2  # m^2
        characteristic_length = side
    elif shape == "Sphere":
        radius = float(radius_entry.get()) * 1e-9  # nm to meters
        volume = (4/3) * np.pi * radius ** 3  # m^3
        surface_area = 4 * np.pi * radius ** 2  # m^2
        characteristic_length = 2 * radius
    elif shape == "Cylinder":
        radius = float(radius_entry.get()) * 1e-9  # nm to meters
        height = float(height_entry.get()) * 1e-9  # nm to meters
        volume = np.pi * radius ** 2 * height  # m^3
        surface_area = 2 * np.pi * radius * (radius + height)  # m^2
        characteristic_length = 2 * radius
    elif shape == "Cuboid" or shape == "Rectangular Prism":
        length = float(length_entry.get()) * 1e-9  # nm to meters
        width = float(width_entry.get()) * 1e-9  # nm to meters
        height = float(height_entry.get()) * 1e-9  # nm to meters
        volume = length * width * height  # m^3
        surface_area = 2 * (length * width + width * height + height * length)  # m^2
        characteristic_length = max(length, width, height)
    
    # Density conversion from g/cm^3 to kg/m^3
    density_kg_m3 = density * 1e3
    mass = volume * density_kg_m3  # kg
    mass_g = mass * 1e3  # g
    
    # Melting Temperature Adjustment
    if shape == "Sphere":
        radius = float(radius_entry.get()) * 1e-9  # nm to meters
    else:
        radius = characteristic_length / 2  # m
    if heat_of_fusion != 0 and radius != 0:
        melting_temperature = melting_point_bulk - (2 * surface_tension) / (heat_of_fusion * radius)
    else:
        melting_temperature = melting_point
    
    # New property calculations
    surface_to_volume_ratio = surface_area / volume  # m^-1
    specific_surface_area = surface_area / mass  # m^2/kg
    
    # Assume a solution volume of 1 mL for particle concentration
    solution_volume = 1e-6  # m^3
    particle_concentration = 1 / volume  # particles/m^3
    
    # Diffusion coefficient (assume water as medium at 25°C)
    temperature = 298.15  # K
    viscosity = 8.9e-4  # Pa·s (for water at 25°C)
    diffusion_coefficient = (const.k * temperature) / (6 * np.pi * viscosity * radius)  # m^2/s
    
    # Sedimentation rate (assume water as medium)
    g = 9.81  # m/s^2
    fluid_density = 1000  # kg/m^3 (for water)
    sedimentation_rate = (2 * radius**2 * g * (density_kg_m3 - fluid_density)) / (9 * viscosity)  # m/s
except Exception as e:
    print(f"Error calculating nanoparticle properties: {e}")

def calculate_debye_length(ionic_strength, temperature):
    try:
        # Implementation of Debye length calculation
        pass
    except Exception as e:
        print(f"Error calculating Debye length: {e}")

def calculate_zeta_potential(radius, charge, temperature):
    try:
        # Function implementation here
        pass
    except Exception as e:
        print(f"Error calculating zeta potential: {e}")

def generate_initial_configuration(shape, dimensions, lattice_constant):
    try:
        # Generate atom positions based on shape and dimensions
        # Return a list of atom positions
        pass
    except Exception as e:
        print(f"Error generating initial configuration: {e}")


def calculate_rdf(positions, bins, max_distance):
    try:
        # Calculate the radial distribution function
        pass
    except Exception as e:
        print(f"Error calculating RDF: {e}")

def calculate_van_der_waals_force(radius1, radius2, hamaker_constant, distance):
    try:
        # Implementation of van der Waals force calculation
        force = (hamaker_constant * (radius1 * radius2)) / (6 * distance ** 2 * (radius1 + radius2))
        return force
    except Exception as e:
        print(f"Error calculating van der Waals force: {e}")
        # Update labels with new properties
        volume_label.config(text=f"Volume: {volume * 1e27:.2f} nm³")  # Convert m^3 to nm^3
        surface_area_label.config(text=f"Surface Area: {surface_area * 1e18:.2f} nm²")  # Convert m^2 to nm^2
        mass_label.config(text=f"Mass: {mass_g:.2e} g")
        material_properties_label.config(
            text=(f"Density: {density:.2f} g/cm³\nRefractive Index: {material_properties[material]['refractive_index']}\n"
                 f"Melting Point: {melting_temperature:.2f} °C\n"
                 f"Surface to Volume Ratio: {surface_to_volume_ratio:.2e} m⁻¹\n"
                 f"Specific Surface Area: {specific_surface_area:.2e} m²/kg\n"
                 f"Particle Concentration: {particle_concentration:.2e} particles/m³\n"
                 f"Diffusion Coefficient: {diffusion_coefficient:.2e} m²/s\n"
                 f"Sedimentation Rate: {sedimentation_rate:.2e} m/s")
                 
            charge = float(self.charge_entry.get())  # Assume this entry exists
            zeta_potential = self.calculate_zeta_potential(radius, charge, temperature)
            # Display results (assume display_results method exists)
            self.display_results(volume, surface_area, mass, melting_temperature, surface_to_volume_ratio, specific_surface_area, diffusion_coefficient, sedimentation_rate, zeta_potential)
        )
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")

    # Clear entry fields after calculating
    side_entry.delete(0, tk.END)
    radius_entry.delete(0, tk.END)
    height_entry.delete(0, tk.END)
    length_entry.delete(0, tk.END)
    width_entry.delete(0, tk.END)

def save_project():
    project_data = {
        "shape": shape_var.get(),
        "material": material_var.get(),
        "category": category_var.get(),
        "side": side_entry.get(),
        "radius": radius_entry.get(),
        "height": height_entry.get(),
        "length": length_entry.get(),
        "width": width_entry.get()
    }
    file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("Text files", "*.txt")])
    if file_path:
        ext = file_path.split('.')[-1]
        if ext == "json":
            with open(file_path, 'w') as file:
                json.dump(project_data, file)
        elif ext == "csv":
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                for key, value in project_data.items():
                    writer.writerow([key, value])
        elif ext == "txt":
            with open(file_path, 'w') as file:
                for key, value in project_data.items():
                    file.write(f"{key}: {value}\n")
        messagebox.showinfo("Save Project", "Project saved successfully.")

def load_project():
    file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("Text files", "*.txt")])
    if file_path:
        ext = file_path.split('.')[-1]
        project_data = {}
        if ext == "json":
            with open(file_path, 'r') as file:
                project_data = json.load(file)
        elif ext == "csv":
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                project_data = {rows[0]: rows[1] for rows in reader}
        elif ext == "txt":
            with open(file_path, 'r') as file:
                for line in file:
                    key, value = line.strip().split(': ')
                    project_data[key] = value

        shape_var.set(project_data["shape"])
        material_var.set(project_data["material"])
        category_var.set(project_data["category"])
        update_materials()
        side_entry.insert(0, project_data["side"])
        radius_entry.insert(0, project_data["radius"])
        height_entry.insert(0, project_data["height"])
        length_entry.insert(0, project_data["length"])
        width_entry.insert(0, project_data["width"])
        shape_changed()
        messagebox.showinfo("Load Project", "Project loaded successfully.")
def calculate_zeta_potential(self, radius, charge, temperature):
        try:
            # Function implementation here
            pass
        except Exception as e:
            print(f"Error calculating zeta potential: {e}")

def generate_initial_configuration(shape, dimensions, lattice_constant):
    try:
        # Generate atom positions based on shape and dimensions
        # Return a list of atom positions
        pass
    except Exception as e:
        print(f"Error generating initial configuration: {e}")

def calculate_rdf(positions, bins, max_distance):
    try:
        # Calculate the radial distribution function
        pass
    except Exception as e:
        print(f"Error calculating RDF: {e}")
def generate_lammps_input():
    # ... existing code ...
    with open(file_path, 'w') as file:
        file.write("# LAMMPS input file for nanoparticle simulation\n")
        file.write(f"units metal\n")
        file.write(f"atom_style charge\n")
        file.write(f"boundary p p p\n")
        file.write(f"read_data nanoparticle.data\n")
        # Add force field parameters
        file.write(f"pair_style lj/cut 10.0\n")
        # Add more LAMMPS commands based on the nanoparticle properties
    
    # Generate separate data file
    with open("nanoparticle.data", 'w') as data_file:
        # Write atom positions, bonds, etc. based on shape and material
        pass

def generate_gromacs_input():
    # ... existing code ...
    with open(file_path, 'w') as file:
        file.write("; GROMACS mdp file for nanoparticle simulation\n")
        file.write("integrator = md\n")
        file.write("dt = 0.001\n")
        file.write("nsteps = 1000000\n")
        # Add more GROMACS parameters
    
    # Generate separate topology file
    with open("nanoparticle.top", 'w') as top_file:
        # Write force field parameters, atom types, etc.
        pass
def visualize_lammps_output():
    file_path = filedialog.askopenfilename(filetypes=[("LAMMPS trajectory files", "*.lammpstrj")])
    if file_path:
        try:
            universe = mda.Universe(file_path, format="LAMMPSDUMP")
            view = nglview.show_mdanalysis(universe)
            display(view)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to visualize LAMMPS output: {str(e)}")

def visualize_gromacs_output():
    tpr_file = filedialog.askopenfilename(filetypes=[("GROMACS tpr files", "*.tpr")])
    xtc_file = filedialog.askopenfilename(filetypes=[("GROMACS xtc files", "*.xtc")])
    if tpr_file and xtc_file:
        try:
            universe = mda.Universe(tpr_file, xtc_file)
            view = nglview.show_mdanalysis(universe)
            display(view)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to visualize GROMACS output: {str(e)}")
def visualize_lammps_output():
    file_path = filedialog.askopenfilename(filetypes=[("LAMMPS trajectory files", "*.lammpstrj")])
    if file_path:
        try:
            universe = mda.Universe(file_path, format="LAMMPSDUMP")
            view = nv.show_mdanalysis(universe)
            view.render_image()
            view.display()
            messagebox.showinfo("Visualization", "LAMMPS visualization displayed. Close the browser tab when done.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to visualize LAMMPS output: {str(e)}")

def visualize_gromacs_output():
    tpr_file = filedialog.askopenfilename(filetypes=[("GROMACS tpr files", "*.tpr")])
    xtc_file = filedialog.askopenfilename(filetypes=[("GROMACS xtc files", "*.xtc")])
    if tpr_file and xtc_file:
        try:
            universe = mda.Universe(tpr_file, xtc_file)
            view = nv.show_mdanalysis(universe)
            view.render_image()
            view.display()
            messagebox.showinfo("Visualization", "GROMACS visualization displayed. Close the browser tab when done.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to visualize GROMACS output: {str(e)}")
def enhanced_3d_visualization():
    shape = shape_var.get()
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        if shape == "Sphere":
            radius = float(radius_entry.get())
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = radius * np.outer(np.cos(u), np.sin(v))
            y = radius * np.outer(np.sin(u), np.sin(v))
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

            # Surface plot for the sphere
            surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, alpha=0.8)

            # Charge distribution visualization (simplified)
            charge = 1.0  # Assume a unit positive charge for visualization
            charge_colors = np.full(z.shape, charge)
            ax.scatter(x, y, z, c=charge_colors, cmap=cm.viridis, s=10, alpha=0.5)

            # Potential field visualization (simplified)
            r = np.sqrt(x**2 + y**2 + z**2)
            potential = charge / r
            max_potential = np.max(potential)
            ax.contour(x, y, z, potential/max_potential, cmap=cm.plasma, alpha=0.3)

        elif shape == "Cube":
            side = float(side_entry.get())
            vertices = np.array([[0, 0, 0], [0, 0, side], [0, side, 0], [0, side, side],
                                 [side, 0, 0], [side, 0, side], [side, side, 0], [side, side, side]])
            ax.scatter3D(vertices[:, 0], vertices[:, 1], vertices[:, 2])

            for s, e in combinations(range(8), 2):
                if np.sum(np.abs(vertices[s] - vertices[e])) == side:
                    ax.plot3D(*zip(vertices[s], vertices[e]), color="b")

            # Simplified charge distribution
            charge_points = np.random.rand(100, 3) * side
            ax.scatter(charge_points[:, 0], charge_points[:, 1], charge_points[:, 2], 
                       c=np.random.rand(100), cmap=cm.viridis, s=20, alpha=0.5)

        # Add more shape visualizations as needed

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{shape} Nanoparticle Visualization')
        
        plt.colorbar(surf, ax=ax, label='Surface Charge', shrink=0.5, aspect=5)
        plt.show()

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")
def plot_3d():
    shape = shape_var.get()
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if shape == "Cube":
            side = float(side_entry.get())
            r = [0, side]
            vertices = [[x, y, z] for x in r for y in r for z in r]
            faces = [
                [vertices[0], vertices[1], vertices[3], vertices[2]],
                [vertices[4], vertices[5], vertices[7], vertices[6]], 
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[1], vertices[3], vertices[7], vertices[5]],
                [vertices[0], vertices[2], vertices[6], vertices[4]]
            ]
            ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
        elif shape == "Sphere":
            radius = float(radius_entry.get())
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = radius * np.outer(np.cos(u), np.sin(v))
            y = radius * np.outer(np.sin(u), np.sin(v))
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='cyan')
        elif shape == "Cylinder":
            radius = float(radius_entry.get())
            height = float(height_entry.get())
            x = np.linspace(-radius, radius, 100)
            z = np.linspace(0, height, 100)
            X, Z = np.meshgrid(x, z)
            Y = np.sqrt(radius ** 2 - X ** 2)
            ax.plot_surface(X, Y, Z, color='cyan')
            ax.plot_surface(X, -Y, Z, color='cyan')
        elif shape == "Cuboid" or shape == "Rectangular Prism":
            length = float(length_entry.get())
            width = float(width_entry.get())
            height = float(height_entry.get())
            r = [0, length]
            l = [0, width]
            h = [0, height]
            vertices = [[x, y, z] for x in r for y in l for z in h]
            faces = [
                [vertices[0], vertices[1], vertices[3], vertices[2]],
                [vertices[4], vertices[5], vertices[7], vertices[6]], 
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[1], vertices[3], vertices[7], vertices[5]],
                [vertices[0], vertices[2], vertices[6], vertices[4]]
            ]
            ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
        plt.show()
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")

def generate_lammps_input():
    try:
        shape = shape_var.get()
        material = material_var.get()
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("LAMMPS files", "*.txt")])
        if file_path:
            with open(file_path, 'w') as file:
                file.write("# LAMMPS input file generated by Tkinter application\n")
                file.write(f"# Shape: {shape}\n")
                file.write(f"# Material: {material}\n")
                # Add more LAMMPS specific parameters here based on calculated properties
            messagebox.showinfo("LAMMPS Input", "LAMMPS input file generated successfully.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def generate_gromacs_input():
    try:
        shape = shape_var.get()
        material = material_var.get()
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("GROMACS files", "*.txt")])
        if file_path:
            with open(file_path, 'w') as file:
                file.write("; GROMACS input file generated by Tkinter application\n")
                file.write(f"; Shape: {shape}\n")
                file.write(f"; Material: {material}\n")
                # Add more GROMACS specific parameters here based on calculated properties
            messagebox.showinfo("GROMACS Input", "GROMACS input file generated successfully.")
    except Exception as e:
        messagebox.showerror("Error", str(e))
def generate_simulation_inputs():
    shape = shape_var.get()
    material = material_var.get()
    
    # Create a directory for the simulation files
    sim_dir = f"{shape}_{material}_simulation"
    os.makedirs(sim_dir, exist_ok=True)
    
    # Generate energy minimization input
    with open(os.path.join(sim_dir, "minimize.in"), 'w') as f:
        f.write("# Energy minimization for nanoparticle\n")
        f.write("clear\n")
        f.write("units metal\n")
        f.write("dimension 3\n")
        f.write("boundary p p p\n")
        f.write("atom_style full\n")
        f.write(f"read_data {shape}_{material}.data\n")
        f.write("pair_style lj/cut 10.0\n")
        f.write("pair_coeff * * 0.1 3.0\n")
        f.write("minimize 1.0e-4 1.0e-6 1000 10000\n")
        f.write("write_data minimized.data\n")
    
    # Generate equilibration input
    with open(os.path.join(sim_dir, "equilibrate.in"), 'w') as f:
        f.write("# Equilibration for nanoparticle\n")
        f.write("clear\n")
        f.write("units metal\n")
        f.write("dimension 3\n")
        f.write("boundary p p p\n")
        f.write("atom_style full\n")
        f.write("read_data minimized.data\n")
        f.write("pair_style lj/cut 10.0\n")
        f.write("pair_coeff * * 0.1 3.0\n")
        f.write("fix 1 all nvt temp 300.0 300.0 100.0\n")
        f.write("timestep 0.001\n")
        f.write("thermo 1000\n")
        f.write("run 100000\n")
        f.write("write_data equilibrated.data\n")
    
    # Generate production run input
    with open(os.path.join(sim_dir, "production.in"), 'w') as f:
        f.write("# Production run for nanoparticle\n")
        f.write("clear\n")
        f.write("units metal\n")
        f.write("dimension 3\n")
        f.write("boundary p p p\n")
        f.write("atom_style full\n")
        f.write("read_data equilibrated.data\n")
        f.write("pair_style lj/cut 10.0\n")
        f.write("pair_coeff * * 0.1 3.0\n")
        f.write("fix 1 all nvt temp 300.0 300.0 100.0\n")
        f.write("timestep 0.001\n")
        f.write("thermo 1000\n")
        f.write("dump 1 all custom 1000 trajectory.lammpstrj id type x y z vx vy vz\n")
        f.write("run 1000000\n")
    
    messagebox.showinfo("Simulation Inputs", f"Simulation input files generated in '{sim_dir}' directory.")

root = tk.Tk()
root.title("Nanoparticle Property Calculator")

shape_var = tk.StringVar()
material_var = tk.StringVar()
shape_var.trace("w", shape_changed)

shapes = ["Cube", "Sphere", "Cylinder", "Cuboid", "Rectangular Prism"]
tk.Label(root, text="Shape:").grid(row=0, column=0, sticky="e")
shape_menu = tk.OptionMenu(root, shape_var, *shapes)
shape_menu.grid(row=0, column=1)

# Create a dictionary to store materials by category
materials_by_category = {}
for material, properties in material_properties.items():
    category = properties['category']
    if category not in materials_by_category:
        materials_by_category[category] = []
    materials_by_category[category].append(material)

# Create StringVar for category and material
category_var = tk.StringVar()
material_var = tk.StringVar()

def update_materials(*args):
    material_menu['menu'].delete(0, 'end')
    for material in materials_by_category[category_var.get()]:
        material_menu['menu'].add_command(label=material, command=tk._setit(material_var, material))
    material_var.set(materials_by_category[category_var.get()][0])

tk.Label(root, text="Category:").grid(row=1, column=0, sticky="e")
category_menu = tk.OptionMenu(root, category_var, *materials_by_category.keys(), command=update_materials)
category_menu.grid(row=1, column=1)

tk.Label(root, text="Material:").grid(row=2, column=0, sticky="e")
material_menu = tk.OptionMenu(root, material_var, "")
material_menu.grid(row=2, column=1)

# Set initial values
category_var.set(list(materials_by_category.keys())[0])
update_materials()

tk.Label(root, text="Side (nm):").grid(row=3, column=0, sticky="e")
side_entry = tk.Entry(root)
side_entry.grid(row=3, column=1)

tk.Label(root, text="Radius (nm):").grid(row=4, column=0, sticky="e")
radius_entry = tk.Entry(root)
radius_entry.grid(row=4, column=1)

tk.Label(root, text="Height (nm):").grid(row=5, column=0, sticky="e")
height_entry = tk.Entry(root)
height_entry.grid(row=5, column=1)

tk.Label(root, text="Length (nm):").grid(row=6, column=0, sticky="e")
length_entry = tk.Entry(root)
length_entry.grid(row=6, column=1)

tk.Label(root, text="Width (nm):").grid(row=7, column=0, sticky="e")
width_entry = tk.Entry(root)
width_entry.grid(row=7, column=1)
tk.Label(root, text="Temperature (K):").grid(row=19, column=0, sticky="e")
temperature_entry = tk.Entry(root)
temperature_entry.grid(row=19, column=1)

tk.Label(root, text="Ionic Strength (mol/L):").grid(row=20, column=0, sticky="e")
ionic_strength_entry = tk.Entry(root)
ionic_strength_entry.grid(row=20, column=1)

# Add more entries for other simulation parameters

tk.Button(root, text="Calculate", command=calculate).grid(row=8, column=0, columnspan=2)

volume_label = tk.Label(root, text="Volume:")
volume_label.grid(row=9, column=0, columnspan=2)

surface_area_label = tk.Label(root, text="Surface Area:")
surface_area_label.grid(row=9, column=0, columnspan=2)

mass_label = tk.Label(root, text="Mass:")
mass_label.grid(row=10, column=0, columnspan=2)
# Add this button to your GUI
tk.Button(root, text="Enhanced 3D Visualization", command=enhanced_3d_visualization).grid(row=21, column=0, columnspan=2)
tk.Button(root, text="Generate Simulation Inputs", command=generate_simulation_inputs).grid(row=22, column=0, columnspan=2)

# Replace the existing material_properties_label with this:
material_properties_label = tk.Label(root, text="Material Properties:", justify=tk.LEFT)
material_properties_label.grid(row=11, column=0, columnspan=2, sticky="w")
tk.Button(root, text="Save Project", command=save_project).grid(row=12, column=0, columnspan=2)
tk.Button(root, text="Load Project", command=load_project).grid(row=13, column=0, columnspan=2)
tk.Button(root, text="Plot 3D", command=plot_3d).grid(row=14, column=0, columnspan=2)
tk.Button(root, text="Generate LAMMPS Input", command=generate_lammps_input).grid(row=15, column=0, columnspan=2)
tk.Button(root, text="Generate GROMACS Input", command=generate_gromacs_input).grid(row=16, column=0, columnspan=2)
tk.Button(root, text="Visualize LAMMPS Output", command=visualize_lammps_output).grid(row=17, column=0, columnspan=2)
tk.Button(root, text="Visualize GROMACS Output", command=visualize_gromacs_output).grid(row=18, column=0, columnspan=2)
shape_var.set(shapes[0])
material_var.set(material[0])
shape_changed()

root.mainloop()
