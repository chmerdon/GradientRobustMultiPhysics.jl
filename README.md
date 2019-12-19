# juliaFE
FiniteElements for Julia

Learning julia by implementing finite elements, don't expect

Features/Limitations:
- many Finite Elements (P0, P1, P2, Taylor--Hood, Mini-Element, P2-Bubble element, Crouzeix-Raviart)
  given through a set of basis functions in reference coordinates and some local2global map
- running solver for Poisson problems, L2 bestapproximation with Dirichlet boundary data
- running solver for incompressible Stokes problems with Dirichlet boundary data
- gradients of FE basis functions also computable by ForwardDiff
- own Grid module (only intended for triangles and 1d or 2d domains so far)
- own Quadrature module


