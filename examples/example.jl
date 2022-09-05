using PolicyNLPModels, MatrixEquations, Random, LinearAlgebra, MadNLP, NLPModels

Random.seed!(0)
# Random.seed!()

nx = 3
nu = 3
nξ = 3

M  = 30
N  = 50
γ  = 0.99
λ  = 1.

A = [
    .9 -.05 0
    -.05 .9 -.05 
    0 -.05 .9 
]
B = [
    1 0 0
    0 1 0
    0 0 1
]
C = I
P = ared(A,B,I,I)[1]
K = -inv(I+B'*P*B)*B'*P*A

rew(x,u,ξ) = (1/2)*dot(x,x) + (1/2)*dot(u,u)
dyn(x,u,ξ) = A*x + B*u + C*ξ
sigm(x) = x/(1+exp(-x))
pol = DensePolicy(tanh, [nx,nx,nx,nu], K)
x0di() = 0.05 .* (rand(nx).*2 .-1) 
ξdi() = 0.05 .* (rand(nξ).*2 .-1)
function con(x, u, ξ)
    [
        x[1]-x[2];
        x[2]-x[3];
        u[1];
        u[2];
        u[3];
    ]
end
nc =  5

gl =-.1*ones(nc)
gu = .1*ones(nc)


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
    # tol = 1e-4,
    # max_iter=1,
    linear_solver=LapackCPUSolver,
    # nlp_scaling = false,
    # print_level = MadNLP.TRACE
)
MadNLP.solve!(solver)
solver2 = solver


Random.seed!(1)

Tsim = 100
Nsam = 10

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
mpc = MPCPolicy(
    N,
    nx,
    nu,
    nξ,
    rew,
    dyn,
    γ = γ,
    con = con,
    gl = gl,
    gu = gu,
    nc = nc,
    jacobian_constant = true,
    hessian_constant = true,
    print_level=MadNLP.ERROR
)


rew_pol, cvio_pol = performance(
    rew,
    dyn,
    x->pol(W,x),
    con,
    gl * 1.0001,
    gu * 1.0001,
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
    gl * 1.0001,
    gu * 1.0001,
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
    mpc,
    con,
    gl * 1.0001,
    gu * 1.0001,
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
