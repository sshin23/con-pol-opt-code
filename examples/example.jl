using PolicyNLPModels, MatrixEquations, Random, LinearAlgebra, MadNLP, NLPModels, LaTeXStrings
using Plots
using JuMP, MadNLPHSL

Random.seed!(1)

function MPCPolicy(mpc,xind,uind,nx,nu,x0)
    mpc.cnt.k = 0
    # fix.(m[:x][:,1],x)
    # optimize!(m)
    # value.(m[:u][:,1])
    mpc.rhs[yind] .= x0
    mpc.x[xind] .= x0
    solve!(mpc)
    mpc.status == MadNLP.SOLVE_SUCCEEDED ? print("/") : print("*") 
    return mpc.status == MadNLP.SOLVE_SUCCEEDED ? mpc.x[uind] : min.(max.(K*x0,ul),uu)
end



Tsim = 100
Nsam = 100

Random.seed!(1)

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
    1. 0 0
    0 1 0
    0 0 1
]

S = [
    0. 0 0 0 0 
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
    -.1
    -.1
    -.1
]
uu = [
    .1
    .1
    .1
]
gl = [
    -.10
    -.20
    -.20
    ul
]
gu = [
    .10
    .20
    .20
    uu
]

P = ared(sqrt(γ)*A,B,R/γ,Q,S')[1]
K = -inv(R+ γ * B'*P*B)*(γ * B'*P*A + S)

nx = size(A,1)
nu = size(B,2)
nξ = size(C,2)
nc = size(E,1)
rew(x,u,ξ) = (1/2)*dot(x,Q,x) + (1/2)*dot(u,R,u)+ dot(u,S,x) + dot(u,T,ξ)
dyn(x,u,ξ) = A*x + B*u + C*ξ
sigm(x) = x/(1+exp(-x))
pol = DensePolicy(tanh, [nx,nx,nu], K)
x0di() = 0.05 .* (rand(nx).*2 .-1) 
ξdi() = 1 .* (rand(nξ).*2 .-1)
function con(x, u, ξ)
    return E*x + F*u
end

Nmpc = 20

m = Model(MadNLP.Optimizer)

set_optimizer_attribute(m, "linear_solver", Ma27Solver)
set_optimizer_attribute(m, "max_iter", 30)
# set_optimizer_attribute(m, "print_level", MadNLP.ERROR)

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


ξs = [zeros(nξ,Tsim) for k=1:Nsam]
x0s= [zeros(nx) for k=1:Nsam]
for k=1:Nsam
    x0s[k] = x0di()
    for i=1:Tsim-1
        ξs[k][:,i] = ξdi()
    end
end

rew_lqr, cvio_lqr = performance(
    rew,
    dyn,
    x->min.(max.(K*x,ul),uu),
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
    x-> MPCPolicy(mpc,xind,uind,nx,nu,x),
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

save = []
# for M in [5 10 15 20]
#     for N in [5 10 15 20]

(M,N) = (20,20)


        Random.seed!(1)

        x0st= [x0di() for i=1:M]
        ξst= [[ξdi() for k=1:N] for i=1:M]

        pol = DensePolicy(tanh, [nx,4,4,nu], K)
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


        nlp.meta.x0 .= randn(PolicyNLPModels.get_W_dim(pol)) * 1.


        nlp.meta.x0[1:length(K)] .= K[:]
        nlp.meta.x0[length(K)+1:end] .= randn(length(nlp.meta.x0) - length(K))

        solver = MadNLPSolver(
            nlp;
            # kkt_system=MadNLP.DENSE_KKT_SYSTEM,
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


        println("$M $N")
        println([rew_pol rew_lqr rew_mpc])
        println([cvio_pol cvio_lqr cvio_mpc])

        push!(save,(rew_pol,cvio_pol))

        if (N,M) == (20,20)
            xpol = zeros(nx,Tsim)
            xlqr = zeros(nx,Tsim)
            xmpc = zeros(nx,Tsim)

            upol = zeros(nu,Tsim)
            ulqr = zeros(nu,Tsim)
            umpc = zeros(nu,Tsim)

            k = 3
            xpol[:,1] .= x0s[k]
            xlqr[:,1] .= x0s[k]
            xmpc[:,1] .= x0s[k]
            upol[:,1] .= pol(W,xpol[:,1])
            ulqr[:,1] .= min.(max.(K*xlqr[:,1],ul),uu)
            umpc[:,1] .= MPCPolicy(mpc,xind,uind,nx,nu,xmpc[:,1])

            for i=2:Tsim
                xpol[:,i] .= dyn(xpol[:,i-1],upol[:,i-1],ξs[k][:,i-1])
                xlqr[:,i] .= dyn(xlqr[:,i-1],ulqr[:,i-1],ξs[k][:,i-1])
                xmpc[:,i] .= dyn(xmpc[:,i-1],umpc[:,i-1],ξs[k][:,i-1])
                upol[:,i] .= pol(W,xpol[:,i])
                ulqr[:,i] .= min.(max.(K*xlqr[:,i],ul),uu)
                umpc[:,i] .= MPCPolicy(mpc,xind,uind,nx,nu,xmpc[:,i])
            end

            for i=1:3
                plt = plot(
                    size = (500,200),
                    xlim=(0,100),
                    # markershape = :auto,
                    framestyle = :box,
                    ylabel = L"x[%$i]",
                    xlabel = L"t",
                    legend=:none,
                    # legend = :bottomleft,
                    fontfamily="Computer Modern"
                )

                plot!(plt, xpol[i,:], label="DPO")
                plot!(plt, xlqr[i,:], label="LQR", linestyle=:dashdot)
                plot!(plt, xmpc[i,:], label="MPC", linestyle=:dash)
                hline!(plt,[gl[i],gu[i]], linestyle=:dot, color=:black, label = :none)
                savefig(plt,"x$i.pdf")

                plt = plot(
                    size = (500,200),
                    xlim=(0,100),
                    # markershape = :auto,
                    framestyle = :box,
                    ylabel = L"u[%$i]",
                    xlabel = L"t",
                    legend= i==1 ? :topright : :none,
                    # legend = :bottomleft,
                    fontfamily="Computer Modern"
                )
                plot!(plt, upol[i,:], label="DPO", linetype=:steppre)
                plot!(plt, ulqr[i,:], label="LQR", linestyle=:dashdot, linetype=:steppre)
                plot!(plt, umpc[i,:], label="MPC", linestyle=:dash, linetype=:steppre)
                hline!(plt,[gl[i+3],gu[i+3]], linestyle=:dot, color=:black, label = :none)
                savefig(plt,"u$i.pdf")
            end
        end
#     end
# end















# using PolicyNLPModels, MatrixEquations, Random, LinearAlgebra, MadNLP, NLPModels, LaTeXStrings
# using Plots
# using JuMP, MadNLPHSL

# save = []
# for M in [5 10 15 20]
#     for N in [5 10 15 20]




#         x0s= [x0di() for i=1:M]
#         ξs= [[ξdi() for k=1:N] for i=1:M]

#         nlp = PolicyNLPModel(
#             N,
#             x0s,
#             ξs,
#             rew,
#             dyn,
#             pol;
#             γ = γ,
#             λ = λ,
#             nc = nc,
#             con = con,
#             gl = gl,
#             gu = gu
#         )

#         Random.seed!(1)
#         nlp.meta.x0 .= randn(PolicyNLPModels.get_W_dim(pol)) * 0.0


#         nlp.meta.x0[1:length(K)] .= K[:]
#         nlp.meta.x0[length(K)+1:end] .= randn(length(nlp.meta.x0) - length(K))

#         solver = MadNLPSolver(
#             nlp;
#             # kkt_system=MadNLP.DENSE_KKT_SYSTEM,
#             kkt_system=MadNLP.DENSE_CONDENSED_KKT_SYSTEM,
#             lapack_algorithm=MadNLP.CHOLESKY,
#             tol = 1e-4,
#             linear_solver=LapackCPUSolver,
#         )
#         MadNLP.solve!(solver)
#         solver2 = solver


#         Random.seed!(1)

#         Tsim = 100
#         Nsam = 100

#         W = solver.x[1:solver.nlp.meta.nvar]

#         ξs = [zeros(nξ,Tsim) for k=1:Nsam]
#         x0s= [zeros(nx) for k=1:Nsam]
#         for k=1:Nsam
#             x0s[k] = x0di()
#             for i=1:Tsim-1
#                 ξs[k][:,i] = ξdi()
#             end
#         end

#         Nmpc = 20

#         m = Model(MadNLP.Optimizer)

#         set_optimizer_attribute(m, "linear_solver", Ma27Solver)
#         set_optimizer_attribute(m, "max_iter", 20)
#         set_optimizer_attribute(m, "print_level", MadNLP.ERROR)

#         @variable(m, x[1:nx,1:Nmpc])
#         @variable(m, u[1:nu,1:Nmpc])

#         @objective(m, Min,
#                    sum(γ^(t-1)* x[i,t]' * Q[i,j] * x[j,t] for i=1:nx for j=1:nx for t=1:Nmpc)
#                    + sum(γ^(t-1)* u[i,t]' * R[i,j] * u[j,t] for i=1:nu for j=1:nu for t=1:Nmpc) 
#                    + sum(2*γ^(t-1)* u[i,t]' * S[i,j] * x[j,t] for i=1:nu for j=1:nx for t=1:Nmpc) )
#         # fix.(x[:,1], 0)
#         @constraint(m, [i=1:nx], x[i,1] == 1e-9*i)
#         @constraint(m, [i=1:nx,t=1:Nmpc-1], x[i,t+1] == sum(A[i,j]*x[j,t] for j=1:nx) + sum(B[i,j]*u[j,t] for j=1:nu))
#         @constraint(m, [i=1:nc,t=1:Nmpc], gl[i] <= sum(E[i,j]*x[j,t] for j=1:nx) + sum(F[i,j]*u[j,t] for j=1:nu) <= gu[i])

#         optimize!(m)
#         mpc = m.moi_backend.optimizer.model.nlp.model.solver

#         yind = [findfirst(mpc.rhs .== 1e-9*i) for i=1:nx]
#         xind = [findfirst(mpc.x .== value(x[i])) for i=1:nx]
#         uind = [findfirst(mpc.x .== value(u[i])) for i=1:nu]

#         function MPCPolicy(mpc,xind,uind,nx,nu,N,x0)
#             mpc.cnt.k = 0
#             # fix.(m[:x][:,1],x)
#             # optimize!(m)
#             # value.(m[:u][:,1])
#             mpc.rhs[yind] .= x0
#             mpc.x[xind] .= x0
#             solve!(mpc)
#             mpc.status == MadNLP.SOLVE_SUCCEEDED ? print("/") : print("*") 
#             return mpc.status == MadNLP.SOLVE_SUCCEEDED ? mpc.x[uind] : min.(max.(K*x0,ul),uu)
#         end

#         rew_pol, cvio_pol = performance(
#             rew,
#             dyn,
#             x->pol(W,x),
#             con,
#             gl * 1.01,
#             gu * 1.01,
#             x0s,
#             ξs,
#             γ,
#             nx,
#             nu,
#             Tsim,
#             Nsam
#         )

#         rew_lqr, cvio_lqr = performance(
#             rew,
#             dyn,
#             x->min.(max.(K*x,ul),uu),
#             con,
#             gl * 1.01,
#             gu * 1.01,
#             x0s,
#             ξs,
#             γ,
#             nx,
#             nu,
#             Tsim,
#             Nsam
#         )


#         rew_mpc, cvio_mpc = performance(
#             rew,
#             dyn,
#             x-> MPCPolicy(mpc,xind,uind,nx,nu,N,x),
#             con,
#             gl * 1.01,
#             gu * 1.01,
#             x0s,
#             ξs,
#             γ,
#             nx,
#             nu,
#             Tsim,
#             Nsam
#         )

#         println("$M $N")
#         println([rew_pol rew_lqr rew_mpc])
#         println([cvio_pol cvio_lqr cvio_mpc])

#         # if (M,N) == (20,20)

#         xpol = zeros(nx,Tsim)
#         xlqr = zeros(nx,Tsim)
#         xmpc = zeros(nx,Tsim)

#         upol = zeros(nu,Tsim)
#         ulqr = zeros(nu,Tsim)
#         umpc = zeros(nu,Tsim)

#         k = 2
#         xpol[:,1] .= x0s[k]
#         xlqr[:,1] .= x0s[k]
#         xmpc[:,1] .= x0s[k]
#         upol[:,1] .= pol(W,xpol[:,1])
#         ulqr[:,1] .= min.(max.(K*xlqr[:,1],ul),uu)
#         umpc[:,1] .= MPCPolicy(mpc,xind,uind,nx,nu,N,xmpc[:,1])

#         for i=2:Tsim
#             xpol[:,i] .= dyn(xpol[:,i-1],upol[:,i-1],ξs[k][:,i-1])
#             xlqr[:,i] .= dyn(xlqr[:,i-1],ulqr[:,i-1],ξs[k][:,i-1])
#             xmpc[:,i] .= dyn(xmpc[:,i-1],umpc[:,i-1],ξs[k][:,i-1])
#             upol[:,i] .= pol(W,xpol[:,i])
#             ulqr[:,i] .= min.(max.(K*xlqr[:,i],ul),uu)
#             umpc[:,i] .= MPCPolicy(mpc,xind,uind,nx,nu,N,xmpc[:,i])
#         end

#         push!(save,(rew_pol,cvio_pol))

#         if (N,M) == (20,20)
#             for i=1:3
#                 plt = plot(
#                     size = (500,200),
#                     xlim=(0,100),
#                     # markershape = :auto,
#                     framestyle = :box,
#                     ylabel = L"x[%$i]",
#                     xlabel = L"t",
#                     legend=:none,
#                     # legend = :bottomleft,
#                     fontfamily="Computer Modern"
#                 )

#                 plot!(plt, xpol[i,:], label="DPO")
#                 plot!(plt, xlqr[i,:], label="LQR", linestyle=:dashdot)
#                 plot!(plt, xmpc[i,:], label="MPC", linestyle=:dash)
#                 hline!(plt,[gl[i],gu[i]], linestyle=:dot, color=:black, label = :none)
#                 savefig(plt,"x$i.pdf")

#                 plt = plot(
#                     size = (500,200),
#                     xlim=(0,100),
#                     # markershape = :auto,
#                     framestyle = :box,
#                     ylabel = L"u[%$i]",
#                     xlabel = L"t",
#                     legend= i==1 ? :topright : :none,
#                     # legend = :bottomleft,
#                     fontfamily="Computer Modern"
#                 )
#                 plot!(plt, upol[i,:], label="DPO")
#                 plot!(plt, ulqr[i,:], label="LQR", linestyle=:dashdot)
#                 plot!(plt, umpc[i,:], label="MPC", linestyle=:dash)
#                 hline!(plt,[gl[i+3],gu[i+3]], linestyle=:dot, color=:black, label = :none)
#                 savefig(plt,"u$i.pdf")
#             end
#         end
#     end
# end
