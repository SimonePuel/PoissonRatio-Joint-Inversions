# Material Properties Geodetic Inversion and Joint Inversion for Fault Slip and Medium Heterogeneity


This repository contains the deterministic adjoint-based inversions of coseismic surface geodetic data for heterogeneous material elastic properties (Poisson's ratio and shear modulus), and a joint inversion for both the fault slip distribution and subduction zone structure. We used a second-order stress accurate, stress-displacement-rotation mixed finite-element elastic formulation including a new infinite-dimension-consistent fault discontinuity implementation (Puel et al., 2022).

We used the open-source libraries, [FEniCS](https://fenicsproject.org) version 2019.1.0 (Logg \& Wells, 2010; Logg et al., 2012) and [hIPPYlib](https://hippylib.github.io) version 3.0.0 (Villa et al., 2016, 2018, 2021) for the forward and adjoint-based inverse modeling, respectively.


### Installation of FEniCS and hIPPYlib open-source libraries

For the installation of the two open-source libraries and all their dependeciens, please look at the file ``FEniCS_hIPPYlib_installation.md``.


### Mesh Generation and Visualization

For models with complex geometries, the open-source mesh geration software, [Gmsh](https://www.gmsh.info/) (Geuzaine \& Remacle, 2009), can be used to create the mesh which is then imported to FEniCS.

[Matplotlib](https://matplotlib.org) and [Paraview](https://www.paraview.org/) can be used to visualize the results in 2D and 3D, repsectively.


### To acknowledge use of these codes, please cite the following publications: 

We distribute these codes free of charge with the hope that you may find it useful in your own research and educational pursuits. In the normal scientific practice, we request that you recognize the efforts of the authors by citing appropriate peer-reviewed paper(s) in presentations and publications (see list below) and we welcome opportunities for collaboration. Please feel free to reach out with any questions you may have about the software or its applications.

- Simone Puel, Thorsten W. Becker, Umberto Villa, Omar Ghattas, Dunyu Liu (2024). "An adjoint-based optimization method for jointly inverting heterogeneous material properties and fault slip from earthquake surface deformation data", _Geophysical Journal International_, **236**(2), pp. 778--797, DOI: https://doi.org/10.1093/gji/ggad442.

- Simone Puel, Eldar Khattatov, Umberto Villa, Dunyu Liu, Omar Ghattas, Thorsten W. Becker (2022). "A mixed, unified forward/inverse framework for earthquake problems: fault implementation and coseismic slip estimate", _Geophysical Journal International_, **230**(2), pp. 733--758, DOI: https://doi.org/10.1093/gji/ggac050.
