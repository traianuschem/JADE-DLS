JADE-DLS: Jupyter-based Angular Dependent Evaluator for Dynamic Light Scattering

JADE-DLS is a python-script embedded in jupyter to determine the hydrodynamic radius with data from dynamic light scattering (DLS). 
As input it uses the normalized time autocorrelation function subtracted by one, g(2)(τ)−1. Multiple methods are used to fit g(2)(τ)−1 
to yield the decay rate Γ at multiple angles. Γ is then plotted against the squared scattering vector q^2 to determine the diffusion 
coefficient D by linear regression. With other experimental data, such as temperature, viscosity etc., the hydrodynamic radius is 
then calculated.

In the current version .ASC-files from ALV Software are used.

A more detailed description of the script will be uploaded in the future.


To build in:
Weitere mögliche Optimierungen (für später):
Numba JIT: Kompiliert Python zu C (~2-3x schneller)
CuPy GPU: Für sehr große Datensätze (~10-100x schneller)
Cython: Kritische Funktionen in C schreiben (~5x schneller)
