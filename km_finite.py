import numpy as np
import matplotlib.pyplot as plt
from pythtb import tb_model

def create_finite_kane_mele(nx, ny, t1=1.0, spin_orb=0.3, rashba=0.25, esite=1.0):
    """
    Create a finite-size Kane-Mele model with open boundary conditions in both directions.

    Parameters:
    - nx, ny: Integer. Number of unit cells along x and y directions.
    - t1: Float. Nearest-neighbor hopping strength.
    - spin_orb: Float. Intrinsic spin-orbit coupling strength.
    - rashba: Float. Rashba spin-orbit coupling strength.
    - esite: Float. On-site energy difference between the two sublattices.
    
    Returns:
    - model: A pythtb tight-binding model object cut to finite size.
    """
    # Lattice vectors
    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    # Orbital positions
    orb = [[1./3., 1./3.], [2./3., 2./3.]]

    # Create model with spin
    model = tb_model(2, 2, lat, orb, nspin=2)
    r3h = np.sqrt(3.0) / 2.0

    # Pauli spin matrices encoded in 4-vector format
    sigma_x = np.array([0., 1., 0., 0])
    sigma_y = np.array([0., 0., 1., 0])
    sigma_z = np.array([0., 0., 0., 1])
    sigma_a= 0.5*sigma_x-r3h*sigma_y
    sigma_b= 0.5*sigma_x+r3h*sigma_y
    sigma_c=-1.0*sigma_x



    # On-site energies
    model.set_onsite([esite, -esite])

    # Nearest-neighbor hoppings (spin-independent)
    model.set_hop(t1, 0, 1, [0, 0])
    model.set_hop(t1, 0, 1, [0, -1])
    model.set_hop(t1, 0, 1, [-1, 0])

    # Intrinsic spin-orbit coupling (next-nearest-neighbor)
    model.set_hop(-1.j * spin_orb * sigma_z, 0, 0, [0, 1])
    model.set_hop(1.j * spin_orb * sigma_z, 0, 0, [1, 0])
    model.set_hop(-1.j * spin_orb * sigma_z, 0, 0, [1, -1])
    model.set_hop(1.j * spin_orb * sigma_z, 1, 1, [0, 1])
    model.set_hop(-1.j * spin_orb * sigma_z, 1, 1, [1, 0])
    model.set_hop(1.j * spin_orb * sigma_z, 1, 1, [1, -1])

    # Rashba spin-orbit coupling (nearest-neighbor)

    model.set_hop(1.j*rashba*sigma_a, 0, 1, [ 0, 0], mode="add")
    model.set_hop(1.j*rashba*sigma_b, 0, 1, [-1, 0], mode="add")
    model.set_hop(1.j*rashba*sigma_c, 0, 1, [ 0,-1], mode="add")

    # Make finite (open boundary conditions)
    model = model.cut_piece(nx, 0, glue_edgs=False)
    model = model.cut_piece(ny, 1, glue_edgs=False)

    return model
# Model parameters
nx, ny = 15, 15
t1 = 1.0
spin_orb = -0.24
rashba = 0.05
esite = 0.7
#esite =3

finite_model = create_finite_kane_mele(nx, ny, t1=t1, spin_orb=spin_orb, rashba=rashba, esite=esite)
hamiltonian = finite_model._gen_ham().shape
print(hamiltonian)
evals, evecs = finite_model.solve_all(eig_vectors=True)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(range(len(evals)), evals/t1, 'b.')
plt.xlabel('State Index')
plt.ylabel('Energy')
plt.title('Energy Spectrum of Finite Kane-Mele Model')
plt.grid(True)
plt.tight_layout()
plt.show()
