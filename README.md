# juliaFE
FiniteElements for Julia
=========================

Learning julia by implementing finite elements

Usage examples:
- 1D Poisson problem: see Example_Poisson_Line.jl
- 2D Poisson problem: see Example_Poisson_Lshape.jl
- 2D Stokes problem: see Example_Stokes_Square.jl

Features/Limitations:
- type-treed FiniteElements module (so far P1,P2,MINI,CR included, each in a separate file)
- running solver for Poisson problems, L2 bestapproximation with Dirichlet boundary data, Stokes problem



Next Goals:
- cache values of basis functions in reference coordinantes at quadrature points
- implement first Hdiv FiniteElement (lowest order Raviart-Thomas)


