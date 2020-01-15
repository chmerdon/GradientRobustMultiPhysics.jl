# juliaFE
FiniteElements for Julia

Learning julia by implementing finite elements

Usage examples:
- 1D Poisson problem: see Example_Poisson_Line.jl
- 2D Poisson problem: see Example_Poisson_Lshape.jl

Features/Limitations:
- type-treed FiniteElements module (so far P1,P2,CR included, each in a separate file)
- running solver for Poisson problems, L2 bestapproximation with Dirichlet boundary data

Next Goals:
- reimplement Stokes FiniteElements and Stokes solver
- implement first Hdiv FiniteElement (Raviart-Thomas)
- ...


