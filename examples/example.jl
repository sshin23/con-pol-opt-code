using PolicyNLPModels, MatrixEquations, Random, LinearAlgebra, MadNLP, NLPModels

Random.seed!(0)

nx = 4
nu = 2
nξ = 4
M  = 10
N  = 10
γ  = 0.99

A = [
    .8 -.1 0 0
    -.1 .8 -.1 0
    0 -.1 .8 -.1
    0 0 -.1 .8 
]
B = [
    1 0
    0 0
    0 1
    0 0
]
C = I

rew(x,u,ξ) = (1/2)*dot(x,x) + (1/2)*dot(u,u)
dyn(x,u,ξ) = A*x + B*u + C*ξ
pol = DensePolicy(tanh, [nx,nx+nu,nx+nu,nx+nu,nu])
x0di() = randn(nx)
ξdi() = randn(nξ)

# nc = nx + nu 
nc =  nu 
function con(x, u, ξ)
    u
    # result[1:length(x)] .= x
    # result[length(x)+1:length(x)+length(u)] .= u
end
gl =-.3*ones(nc)
gu = .3*ones(nc)

W = randn(PolicyNLPModels.get_W_dim(pol)) * 0.001
x0s= [x0di() for i=1:M]
ξs= [[ξdi() for k=1:N] for i=1:M]

noise = randn(PolicyNLPModels.get_W_dim(pol)) * 0.1

nlp = PolicyNLPModel(
    N,
    x0s,
    ξs,
    rew,
    dyn,
    pol;
    γ = γ,
    λ = 1.0,
    nc = nc,
    con = con,
    gl = gl,
    gu = gu
)

P = ared(A,B,I,I)[1]
K=-inv(I+B'*P*B)*B'*P*A

nlp.meta.x0[1:length(K)] .= K[:]
nlp.meta.x0[length(K)+1:end] .= randn(length(nlp.meta.x0) - length(K))

solver = MadNLPSolver(
    nlp;
    kkt_system=MadNLP.DENSE_CONDENSED_KKT_SYSTEM,
    tol = 1e-3,
    linear_solver=LapackCPUSolver,
    lapack_algorithm=MadNLP.CHOLESKY
)

MadNLP.solve!(solver)


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
    gl * 1.1,
    gu * 1.1,
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
    gl * 1.1,
    gu * 1.1,
    x0s,
    ξs,
    γ,
    nx,
    nu,
    Tsim,
    Nsam
)


# rew_mpc, cvio_mpc = performance(
#     rew,
#     dyn,
#     mpc,
#     con,
#     gl * 1.1,
#     gu * 1.1,
#     x0s,
#     ξs,
#     γ,
#     nx,
#     nu,
#     Tsim,
#     Nsam
# )

[rew_pol rew_lqr rew_mpc]
