import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# Simulation parameters
KbT = 2
beta = 1/KbT
box_size = 3  # Size of the cubic simulation box
num_steps = 2000
sigma = 0.3
cutoff_distance = 2.5 * sigma  # Cutoff distance for the Lennard-Jones potential
molecule_volume = (4/3) * np.pi * sigma**3
epsilons = [0.25,0.5,0.75,1]

# Initialize atom positions randomly within the box

def lennard_jones_potential(r,epsilon):
    r6 = (sigma/r)**6
    r12 = r6 * r6
    return 4 * epsilon*(r12-r6)



def calculate_total_energy(N,positions,epsilon):
  energy = 0
  for i in range(N-1):
    for j in range(i + 1, N):
      r = np.linalg.norm(positions[i] - positions[j])
      if r < cutoff_distance:
        energy += lennard_jones_potential(r,epsilon)
  return energy

# for change in energy as a particle is displaced
def delta_energy(N,old,new,index,positions,epsilon):
  # particle at index is moved from old to new
  energy = 0
  for i in range(N):
    if i == index:
      continue
    r_old = np.linalg.norm(positions[i] - old)
    r_new = np.linalg.norm(positions[i] - new)
    if r_new < cutoff_distance:
      energy += lennard_jones_potential(r_new,epsilon)
    if r_old < cutoff_distance:
      energy -= lennard_jones_potential(r_old,epsilon)
  return energy

def particle_addition_energy(N,new_pos,positions,epsilon):
  # Calculate the energy difference ΔU if we add a particle at  newpos
  delta_U = 0
  for i in range(N):
      r = np.linalg.norm(positions[i] - new_pos)
      if r < cutoff_distance:
          delta_U += lennard_jones_potential(r,epsilon)
  return delta_U

def metropolis_transition(N,positions,epsilon): # does metropolis step and returns change in energy
  index = np.random.randint(N) #random particle
  old_pos = positions[index].copy()
  new_pos = old_pos + (np.random.rand(3) - 0.5) * 0.1
  delta_U = delta_energy(N,old_pos,new_pos,index,positions,epsilon)
  if delta_U < 0 or np.random.rand() < np.exp(-delta_U/KbT): #accept move
    positions[index] = new_pos
    return delta_U
  else: #reject move
    return 0


def calculate_chemical_potential(N,positions,epsilon):
    # Initialize a list to store Boltzmann factors for each sample
    boltzmann_factors = []

    # Generate a random position for the additional particle
    for _ in range(1000):  # Use a large number of samples for averaging
        new_position = np.random.rand(3) * box_size

        # Calculate the energy difference ΔU if we add a particle at this position
        delta_U = particle_addition_energy(N,new_position, positions,epsilon)

        boltzmann_factors.append(np.exp(-beta * delta_U))

    avg_boltzmann_factor = np.mean(boltzmann_factors)

    # Calculate the excess chemical potential
    mu_ex = -KbT * np.log(avg_boltzmann_factor)
    return mu_ex

def radial_distribution_function(N,positions,bin_size=0.1):
    rdf = np.zeros(int(box_size / bin_size))
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < box_size / 2:
                bin_index = int(r / bin_size)
                rdf[bin_index] += 2

    # Normalize the RDF
    r_values = np.arange(len(rdf)) * bin_size;r_values[0] = 0.001
    shell_volumes = 4 * np.pi * (r_values**2) * bin_size
    number_density = N / box_size**3
    with np.errstate(divide='ignore', invalid='ignore'):
        rdf = np.where(shell_volumes > 0, rdf / (shell_volumes * number_density * N), 0)

    return rdf, r_values

def calculate_pressure(N,rdf, r_values):
    # Calculate the pressure using the virial equation
    rho = N / box_size**3
    integral = np.sum((r_values**2) * rdf) * (r_values[1] - r_values[0])
    pressure = rho / beta * (1 + (2 * np.pi * rho / 3) * integral)
    return pressure



# Lists to store results
volume_fractions = [num_atoms * molecule_volume / box_size**3 for num_atoms in range(10,101,5)]
chemical_potentials = []
pressures = []

# Iterate over different numbers of atoms
for epsilon in epsilons:
  mu = []
  P = []
  for num_atoms in range(10,101,5):
    # Initialize atom positions randomly within the box
    positions = np.random.rand(num_atoms, 3) * box_size
    # Equilibriation
    for step in range(num_steps):
      metropolis_transition(num_atoms,positions,epsilon)

    # Calculate chemical potential
    chemical_potential = calculate_chemical_potential(num_atoms,positions,epsilon)
    mu.append(chemical_potential)

    # Calculate RDF and pressure
    rdf, r_values = radial_distribution_function(num_atoms,positions)
    pressure = calculate_pressure(num_atoms,rdf, r_values)
    P.append(pressure)

  # Append results to lists
  chemical_potentials.append(mu)
  pressures.append(P)

# Plotting the results
plt.figure(figsize=(12, 6))

# Plot chemical potential vs volume fraction
plt.subplot(1, 2, 1)
for i in range(len(epsilons)):
  plt.plot(volume_fractions, chemical_potentials[i], marker='o',label = f"ε = {epsilons[i]}")
plt.legend()
plt.xlabel('Volume Fraction')
plt.ylabel('Chemical Potential')
plt.title('Chemical Potential vs Volume Fraction')

# Plot pressure vs volume fraction
plt.subplot(1, 2, 2)
for i in range(len(epsilons)):
  plt.plot(volume_fractions, pressures[i], marker='o',label = f"ε = {epsilons[i]}")
plt.legend()
plt.xlabel('Volume Fraction')
plt.ylabel('Pressure')
plt.title('Pressure vs Volume Fraction')

plt.tight_layout()
plt.show()
