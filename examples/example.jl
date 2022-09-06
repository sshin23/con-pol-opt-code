using PolicyNLPModels, MatrixEquations, Random, LinearAlgebra, MadNLP, NLPModels

Random.seed!(0)
# Random.seed!()

M  = 10
N  = 10
γ  = .99
λ  = .0

### A definition
A11 = [
    .9 -.05 0
    -.05 .9 -.05 
    0 -.05 .9 
]
A12 = [
    0.1 0
    0.2 0
    0.1 0  
]
A22 = exp([0 1; -1 0]*.3)

A = [
    A11 A12
    zeros(size(A12))' A22
]

### B definition
B1 = [
    1 0 0
    0 1 0
    0 0 1
]

B = [B1; zeros(2,3)]

### Q definition
Q1 = [
    1 0 0
    0 2 0
    0 0 1
]

Q = [
    Q1 zeros(size(A12))
    zeros(size(A12))' zeros(size(A22))
]

### R definition
R = [
    2 0 0
    0 1 0
    0 0 1
]

### C definition
C  = [
    1 0
    0 1
    1 1
    0 0
    0 0
]

### E definition
E1 = [
    1 0 0
    0 1 0
    0 0 1
    0 0 0
    0 0 0
    0 0 0
]

E = [
    E1 zeros(size(E1,1),size(A22,1))
]

### F definition
F = [
    0 0 0
    0 0 0
    0 0 0
    1 0 0
    0 1 0
    0 0 1
]

gl = [
    -.30
    -.30
    -.30
    -.03
    -.03
    -.03
]
gu = [
    .30
    .30
    .30
    .03
    .03
    .03
]

P = ared(sqrt(γ)*A,B,R/γ,Q)[1]
K = -γ * inv(R+ γ * B'*P*B)*B'*P*A

nx = size(A,1)
nu = size(B,2)
nξ = size(C,2)
nc = size(E,1)
rew(x,u,ξ) = (1/2)*dot(x,Q,x) + (1/2)*dot(u,R,u)
dyn(x,u,ξ) = A*x + B*u + C*ξ
sigm(x) = x/(1+exp(-x))
pol = DensePolicy(tanh, [nx,nx,nu], K)
x0di() = 0.05 .* (rand(nx).*2 .-1) 
ξdi() = 0.05 .* (rand(nξ).*2 .-1)
function con(x, u, ξ)
    return E*x + F*u
end



x0s= [x0di() for i=1:M]
ξs= [[ξdi() for k=1:N] for i=1:M]

nlp = PolicyNLPModel(
    N,
    x0s,
    ξs,
    rew,
    dyn,
    pol;
    γ = γ,
    λ = λ,
    nc = nc,
    con = con,
    gl = gl,
    gu = gu
)

Random.seed!(1)
nlp.meta.x0 .= randn(PolicyNLPModels.get_W_dim(pol)) * 0.0


nlp.meta.x0[1:length(K)] .= K[:]
nlp.meta.x0[length(K)+1:end] .= randn(length(nlp.meta.x0) - length(K))

# solver = MadNLPSolver(
#     nlp;
#     kkt_system=MadNLP.DENSE_KKT_SYSTEM,
#     # kkt_system=MadNLP.DENSE_CONDENSED_KKT_SYSTEM,
#     # lapack_algorithm=MadNLP.CHOLESKY,
#     # tol = 1e-3,
#     max_iter=1,
#     linear_solver=LapackCPUSolver,
#     nlp_scaling = false,
#     # print_level = MadNLP.TRACE
# )
# MadNLP.solve!(solver)
# solver1 = solver
solver = MadNLPSolver(
    nlp;
    # kkt_system=MadNLP.DENSE_KKT_SYSTEM,
    kkt_system=MadNLP.DENSE_CONDENSED_KKT_SYSTEM,
    lapack_algorithm=MadNLP.CHOLESKY,
    tol = 1e-4,
    # max_iter=1,
    linear_solver=LapackCPUSolver,
    # nlp_scaling = false,
    # print_level = MadNLP.TRACE
)
MadNLP.solve!(solver)
solver2 = solver


Random.seed!(1)

Tsim = 100
Nsam = 100

W = solver.x[1:solver.nlp.meta.nvar]

ξs = [zeros(nξ,Tsim) for k=1:Nsam]
x0s= [zeros(nx) for k=1:Nsam]
for k=1:Nsam
    x0s[k] = x0di()
    for i=1:Tsim-1
        ξs[k][:,i] = ξdi()
    end
end

# mpc
# mpc = MPCPolicy(
#     10,
#     nx,
#     nu,
#     nξ,
#     rew,
#     dyn,
#     γ = γ,
#     con = con,
#     gl = gl,
#     gu = gu,
#     nc = nc,
#     jacobian_constant = true,
#     hessian_constant = true,
#     print_level=MadNLP.ERROR
# )

Nmpc = 20
using JuMP, MadNLPHSL

m = Model(MadNLP.Optimizer)

set_optimizer_attribute(m, "linear_solver", Ma27Solver)
set_optimizer_attribute(m, "max_iter", 10)
set_optimizer_attribute(m, "print_level", MadNLP.ERROR)

@variable(m, x[1:nx,1:Nmpc])
@variable(m, u[1:nx,1:Nmpc])

@objective(m, Min,
           sum(γ^(t-1)* x[i,t]' * Q[i,j] * x[j,t] for i=1:nx for j=1:nx for t=1:Nmpc) + sum(γ^(t-1)* u[i,t]' * R[i,j] * u[j,t] for i=1:nu for j=1:nu for t=1:Nmpc) )
# fix.(x[:,1], 0)
@constraint(m, [i=1:nx], x[i,1] == 1e-9*i)
@constraint(m, [i=1:nx,t=1:Nmpc-1], x[i,t+1] == sum(A[i,j]*x[j,t] for j=1:nx) + sum(B[i,j]*u[j,t] for j=1:nu))
@constraint(m, [i=1:nc,t=1:Nmpc], gl[i] <= sum(E[i,j]*x[j,t] for j=1:nx) + sum(F[i,j]*u[j,t] for j=1:nu) <= gu[i])

optimize!(m)
mpc = m.moi_backend.optimizer.model.nlp.model.solver

yind = [findfirst(mpc.rhs .== 1e-9*i) for i=1:nx]
xind = [findfirst(mpc.x .== value(x[i])) for i=1:nx]
uind = [findfirst(mpc.x .== value(u[i])) for i=1:nu]



function MPCPolicy(mpc,xind,uind,nx,nu,N,x0)
    mpc.cnt.k = 0
    # fix.(m[:x][:,1],x)
    # optimize!(m)
    # value.(m[:u][:,1])
    mpc.rhs[yind] .= x0
    mpc.x[xind] .= x0
    solve!(mpc)
    # mpc.status == 1 ? print("/") : print("*") 
    return mpc.status == MadNLP.SOLVE_SUCCEEDED ? mpc.x[uind] : K*x0
end


rew_pol, cvio_pol = performance(
    rew,
    dyn,
    x->pol(W,x),
    con,
    gl * 1.01,
    gu * 1.01,
    x0s,
    ξs,
    γ,
    nx,
    nu,
    Tsim,
    Nsam
)

rew_lqr, cvio_lqr = performance(
    rew,
    dyn,
    x->K*x,
    con,
    gl * 1.01,
    gu * 1.01,
    x0s,
    ξs,
    γ,
    nx,
    nu,
    Tsim,
    Nsam
)


rew_mpc, cvio_mpc = performance(
    rew,
    dyn,
    x->MPCPolicy(mpc,xind,uind,nx,nu,N,x),
    con,
    gl * 1.01,
    gu * 1.01,
    x0s,
    ξs,
    γ,
    nx,
    nu,
    Tsim,
    Nsam
)

show([rew_pol rew_lqr rew_mpc])
show([cvio_pol cvio_lqr cvio_mpc])

