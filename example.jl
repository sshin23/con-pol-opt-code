include("PolicyOptimization.jl")

using .PolicyOptimization, MatrixEquations, Random, LinearAlgebra, NLPModels, JuMP, MadNLP, MadNLPHSL, LaTeXStrings, Plots


Random.seed!(0)
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
    1e-3 0 0
    0 1e-3 0
    0 0 1e-3
]

Q = [
    Q1 zeros(size(A12))
    zeros(size(A12))' zeros(size(A22))
]

### R definition
R = [
    1 0 0
    0 1 0
    0 0 1
]

S = [
    0 0 0 1 0 
    0 0 0 1 0
    0 0 0 1 0
]

T = [
    .03 
    .03
    .03 
][:,:]

### C definition
C  = [
    .03 
    .03 
    .03 
    0 
    0 
][:,:]

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

ul = [
    -.03
    -.03
    -.03
]
uu = [
    .03
    .03
    .03
]
xl = [
    -.10
    -.10
    -.10
]
xu = [
    .10
    .10
    .10
]
gl = [
    xl
    ul
]
gu = [
    xu
    uu
]

nx = size(A,1)
nu = size(B,2)
nξ = size(C,2)
nc = size(E,1)
rew(x,u,ξ) = (1/2)*dot(x,Q,x) + (1/2)*dot(u,R,u)+ dot(u,S,x) + dot(u,T,ξ)
dyn(x,u,ξ) = A*x + B*u + C*ξ
sigm(x) = x/(1+exp(-x))
x0di() = 0.05 .* (rand(nx).*2 .-1) 
ξdi() = 0 .* (rand(nξ).*2 .-1)
function con(x, u, ξ)
    return E*x + F*u
end

# Generating validation data
Tsim = 100
Nsam = 100

ξs = [zeros(nξ,Tsim) for k=1:Nsam]
x0s= [zeros(nx) for k=1:Nsam]
for k=1:Nsam
    x0s[k] = x0di()
    for i=1:Tsim-1
        ξs[k][:,i] = ξdi()
    end
end

# for LQR
P = ared(sqrt(γ)*A,B,R/γ,Q,S')[1]
K = -inv(R+ γ * B'*P*B)*(γ * B'*P*A + S)

rew_lqr, cvio_lqr = performance(
    rew,
    dyn,
    x->min.(max.(K*x,ul),uu),
    con,
    gl * 1.00,
    gu * 1.00,
    x0s,
    ξs,
    γ,
    nx,
    nu,
    Tsim,
    Nsam
)

# for MPC
Nmpc = 20

m = Model(MadNLP.Optimizer)

if @isdefined(Ma27Solver)
    set_optimizer_attribute(m, "linear_solver", Ma27Solver)
end
set_optimizer_attribute(m, "max_iter", 20)
set_optimizer_attribute(m, "print_level", MadNLP.ERROR)

@variable(m, x[1:nx,1:Nmpc])
@variable(m, u[1:nu,1:Nmpc])

@objective(m, Min,
           sum(γ^(t-1)* x[i,t]' * Q[i,j] * x[j,t] for i=1:nx for j=1:nx for t=1:Nmpc)
           + sum(γ^(t-1)* u[i,t]' * R[i,j] * u[j,t] for i=1:nu for j=1:nu for t=1:Nmpc) 
           + sum(2*γ^(t-1)* u[i,t]' * S[i,j] * x[j,t] for i=1:nu for j=1:nx for t=1:Nmpc) )
# fix.(x[:,1], 0)
@constraint(m, [i=1:nx], x[i,1] == 1e-9*i)
@constraint(m, [i=1:nx,t=1:Nmpc-1], x[i,t+1] == sum(A[i,j]*x[j,t] for j=1:nx) + sum(B[i,j]*u[j,t] for j=1:nu))
@constraint(m, [i=1:nc,t=1:Nmpc], gl[i] <= sum(E[i,j]*x[j,t] for j=1:nx) + sum(F[i,j]*u[j,t] for j=1:nu) <= gu[i])

optimize!(m)
mpc = m.moi_backend.optimizer.model.nlp.model.solver

yind = [findfirst(mpc.rhs .== 1e-9*i) for i=1:nx]
xind = [findfirst(mpc.x .== value(x[i])) for i=1:nx]
uind = [findfirst(mpc.x .== value(u[i])) for i=1:nu]

rew_mpc, cvio_mpc = performance(
    rew,
    dyn,
    x-> MPCPolicy(mpc,xind,uind,yind,nx,nu,x,xl,xu),
    con,
    gl * 1.00,
    gu * 1.00,
    x0s,
    ξs,
    γ,
    nx,
    nu,
    Tsim,
    Nsam
)


save = []

for M in [5 10 15 20]
    for N in [5 10 15 20]


        Random.seed!(1)

        x0st= [x0di() for i=1:M]
        ξst= [[ξdi() for k=1:N] for i=1:M]
        pol = DensePolicy(tanh, [nx,nx,nu,nu], K)

        nlp = PolicyNLPModel(
            N,
            x0st,
            ξst,
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

        nlp.meta.x0 .= randn(length(nlp.meta.x0)) * 0.01

        solver = MadNLPSolver(
            nlp;
            kkt_system=MadNLP.DENSE_CONDENSED_KKT_SYSTEM,
            lapack_algorithm=MadNLP.CHOLESKY,
            tol = 1e-4,
            linear_solver=LapackCPUSolver,
        )
        MadNLP.solve!(solver)


        W = solver.x[1:solver.nlp.meta.nvar]
        

        rew_pol, cvio_pol = performance(
            rew,
            dyn,
            x->pol(W,x),
            con,
            gl * 1.00,
            gu * 1.00,
            x0s,
            ξs,
            γ,
            nx,
            nu,
            Tsim,
            Nsam
        )

        println("$M $N")
        println([rew_pol rew_lqr rew_mpc])
        println([cvio_pol cvio_lqr cvio_mpc])

        push!(save,(rew_pol,cvio_pol))

        if (N,M) == (20,20) # we only plot the last result
            k = 5
            
            xpol = zeros(nx,Tsim)
            xlqr = zeros(nx,Tsim)
            xmpc = zeros(nx,Tsim)

            upol = zeros(nu,Tsim)
            ulqr = zeros(nu,Tsim)
            umpc = zeros(nu,Tsim)

            xpol[:,1] .= x0s[k]
            xlqr[:,1] .= x0s[k]
            xmpc[:,1] .= x0s[k]
            upol[:,1] .= pol(W,xpol[:,1])
            ulqr[:,1] .= min.(max.(K*xlqr[:,1],ul),uu)
            umpc[:,1] .= MPCPolicy(mpc,xind,uind,yind,nx,nu,xmpc[:,1],xl,xu)
            
            for i=2:Tsim
                xpol[:,i] .= dyn(xpol[:,i-1],upol[:,i-1],ξs[k][:,i-1])
                xlqr[:,i] .= dyn(xlqr[:,i-1],ulqr[:,i-1],ξs[k][:,i-1])
                xmpc[:,i] .= dyn(xmpc[:,i-1],umpc[:,i-1],ξs[k][:,i-1])
                upol[:,i] .= pol(W,xpol[:,i])
                ulqr[:,i] .= min.(max.(K*xlqr[:,i],ul),uu)
                umpc[:,i] .= MPCPolicy(mpc,xind,uind,yind,nx,nu,xmpc[:,i],xl,xu)
            end

            for i=1:3
                

                plt = plot(
                    size = (500,120),
                    xlim=(0,100),
                    framestyle = :box,
                    ylabel = L"x[%$i]",
                    xlabel = L"t",
                    legend=:none,
                    fontfamily="Computer Modern"
                )

                plot!(plt, xpol[i,:], label="PO")
                plot!(plt, xlqr[i,:], label="LQR", linestyle=:dashdot)
                plot!(plt, xmpc[i,:], label="MPC", linestyle=:dash)
                hline!(plt,[gl[i],gu[i]], linestyle=:dot, color=:black, label = :none)
                savefig(plt,"x$i.pdf")

                plt = plot(
                    size = (500,120),
                    xlim=(0,100),
                    framestyle = :box,
                    ylabel = L"u[%$i]",
                    xlabel = L"t",
                    legend= i==1 ? :topright : :none,
                    fontfamily="Computer Modern"
                )
                plot!(plt, upol[i,:], label="PO", linetype = :steppre)
                plot!(plt, ulqr[i,:], label="LQR", linestyle=:dashdot, linetype = :steppre)
                plot!(plt, umpc[i,:], label="MPC", linestyle=:dash, linetype = :steppre)
                hline!(plt,[gl[i+3],gu[i+3]], linestyle=:dot, color=:black, label = :none)
                savefig(plt,"u$i.pdf")
            end
        end
    end
end

rew_table = reshape([rew for (rew,cvio) in save],4,4)'
cvio_table = reshape([cvio for (rew,cvio) in save],4,4)'
