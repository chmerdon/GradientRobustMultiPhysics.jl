
# PDE Prototypes

Below all available prototypes (i.e. pre-defined constructors for PDEDescription) are listed.


!!! note

    For most prototypes boundary data and right-hand side data or other modifications to the weak form of the PDE have to be added after a proto-type constructor has been called, see the examples for further assistance.


## Poisson equation

The Poisson equation seeks a function ``u`` such that
```math
- \mu \Delta u = f
```
where ``\mu`` is some diffusion coefficient and ``f`` some given right-hand side data.

The (primal) weak formulation (for homogeneous Dirichlet boundary data) seeks ``u`` such that

```math
(\mu \nabla u,\nabla v)  = (f,v) \quad \text{for all } v\in H^1_0(\Omega)
```

A vanilla PDEDescription for this weak formulation (without boundary data) can be created with the constructor below.

```@docs
PoissonProblem
```

Example-Script: EXAMPLE_ConvectionDiffusion.jl

Remarks:
- dual weak formulations are also possible but are not available as a prototype currently


## Incompressible Navier--Stokes equations

The Navier--Stokes equations in d dimensions seek a (vector-valued) velocity ``\mathbf{u}`` and a pressure ``p`` such that
```math
\begin{aligned}
- \mu \Delta \mathbf{u} + (\mathbf{u} \cdot \nabla) \mathbf{u} + \nabla p & = \mathbf{f}\\
\mathrm{div}(u) & = 0
\end{aligned}
```
where ``\mu`` is some viscosity coefficient and ``f`` some given right-hand side data.

The weak formulation (for homogeneous Dirichlet boundary data) seeks ``(\mathbf{u},p)`` such that

```math
\begin{aligned}
(\mu \nabla \mathbf{u},\nabla \mathbf{v}) + ((u \cdot \nabla) \mathbf{u}, \mathbf{v}) + (\mathrm{div} \mathbf{v}, p) & = (\mathbf{f},\mathbf{v}) && \text{for all } \mathbf{v}\in H^1_0(\Omega)^d\\
(\mathrm{div} \mathbf{u}, q) & = 0 && \text{for all } q \in L^2_0(\Omega)
\end{aligned}
```

A vanilla PDEDescription for this weak formulation (without boundary data) can be created with the constructor below.

```@docs
IncompressibleNavierStokesProblem
```

Remarks:
- if nonlinear == false the nonlinear convection term is not added to the equation resulting in the plain Stokes equations.
- if nopressureconstraint == true removes the integral mean constraint on the pressure.

Example-Scripts: EXAMPLE_Stokes.jl, EXAMPLE_Stokes_probust.jl

## Compressible Navier--Stokes equations


The compressible Navier--Stokes equations in d dimensions seek a (vector-valued) velocity ``\mathbf{u}``, a density ``\varrho`` and a pressure ``p`` such that
```math
\begin{aligned}
- \mu \Delta \mathbf{u} + \lambda \nabla(\mathrm{div}(\mathbf{u}))  + (\mathbf{u} \cdot \nabla) \mathbf{u} + \nabla p & = \mathbf{f} + \varrho \mathbf{g}\\
\mathrm{div}(\varrho \mathbf{u}) & = 0\\
        p & = eos(\varrho)\\
        \int_\Omega \varrho \, dx & = M\\
        \varrho & \geq 0
\end{aligned}
```
where eos is some equation of state function that describes the dependence of the pressure on the density (and further physical quantities like temperature in a more general setting). Moreover, ``\mu`` and ``\lambda`` are Lame parameters and ``\mathbf{f}`` and ``\mathbf{g}`` are given right-hand side data.


The weak formulation (for homogeneous Dirichlet boundary data) seeks ``(\mathbf{u},p,\varrho)`` such that
```math
\begin{aligned}
(\mu \nabla \mathbf{u},\nabla \mathbf{v}) + (\lambda \mathrm{div} \mathbf{u},\mathrm{div} \mathbf{v})  + ((u \cdot \nabla) \mathbf{u}, \mathbf{v}) + (\mathrm{div} \mathbf{v}, p) & = (\mathbf{f},\mathbf{v}) && \text{for all } \mathbf{v} \in H^1_0(\Omega)^d\\
(\varrho \mathbf{u}, \nabla q) & = 0 && \text{for all } q \in W^{1,\infty}(\Omega)\\
        p & = eos(\varrho)\\
        \int_\Omega \varrho \, dx & = M\\
        \varrho & \geq 0
\end{aligned}
```

A vanilla PDEDescription for a proper weak formulation that can be solved iteratively can be generated with the constructor below.


```@docs
CompressibleNavierStokesProblem
```

Example-Script: EXAMPLE_CompressibleStokes.jl



## Navier-Lame equations (linear elasticity)


The Navier-Lame equations seek a displacement ``\mathbf{u}`` such that
```math
- \mathrm{div}( \mathbb{C} \epsilon( \mathbf{u})) = \mathbf{f}
```
where ``\epsilon( \mathbf{u})`` is the symmetric part of the gradient, ``\mathbb{C}`` is the stiffness tensor (according to Hooke's law) and ``\mathbf{f}`` some given right-hand side data.

In 1D, it is assumed that the stiffness tensor has the form
```math
\mathbb{C} \epsilon( u) = \mu \nabla u
```
where ``\mu`` is the elasticity modulus.
In 2D, it is assumed that the stiffness tensor has the form
```math
\mathbb{C} \epsilon( u) = 2 \mu \epsilon( \mathbf{u}) + \lambda \mathrm{tr}(\epsilon( \mathbf{u}))
```
where ``\mu`` and ``\lambda`` are the Lame coefficients.


The (primal) weak formulation (for homogeneous Dirichlet boundary data) seeks ``u`` such that
```math
(\mathbb{C} \epsilon(\mathbf{u}),\epsilon(\mathbf{v})) = (\mathbf{f},\mathbf{v}) \quad \text{for all } v\in H^1_0(\Omega)^d
```

A vanilla PDEDescription for this weak formulation (without boundary data) can be created with the constructor below.

```@docs
LinearElasticityProblem
```

Example-Scripts: EXAMPLE_CookMembrane.jl, EXAMPLE_ElasticTire2.jl


## L2-Bestapproximation

This PDEDescription can be used to setup an L2-Bestapproximation very fast. The weak formulation simply seeks some function ``u`` such that, for some given function ``u_\text{exact}``, it holds ``u = u_\text{exact}`` along the (specified) boundary and

```math
(u,v) = (u_\text{exact},v) \quad \text{for all } v\in L^2(\Omega)
```

Of course, on the continuous level, it holds ``u = u_\text{exact}``, but if the weak formulation is assembled for a finite element space one obtains a discrete L2-bestapproximation for this space.


```@docs
L2BestapproximationProblem
```


## H1-Bestapproximation

This PDEDescription can be used to setup an H1-Bestapproximation very fast. The weak formulation simply seeks some function ``u`` such that, for some given function ``u_\text{exact}``, it holds ``u = u_\text{exact}`` along the (specified) boundary and

```math
(\nabla u,\nabla v) = (\nabla u_\text{exact}, \nabla v) \quad \text{for all } v\in H^1_0(\Omega)
```



```@docs
H1BestapproximationProblem
```