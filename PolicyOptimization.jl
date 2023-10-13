using ForwardDiff, LinearAlgebra, NLPModels, MadNLP, Random, Statistics

export PolicyNLPModel, DensePolicy, performance, MPCPolicy

struct PolicyNLPModel{T, V <: AbstractVector{T}, R, D, P, C} <: AbstractNLPModel{T, Vector{T}}
    N::Int
    nc::Int
    x0s::Vector{V}
    ξs::Vector{Vector{V}}
    γ::T
    λ::T
    rew::R
    dyn::D
    pol::P
    con::C
    g::V
    meta::AbstractNLPModelMeta{T, V}
    counters::NLPModels.Counters
end

function PolicyNLPModel(
    N,
    x0s::Vector{V},
    ξs::Vector{Vector{V}},
    rew::R,
    dyn::D,
    pol::P;
    γ::T = 1.,
    λ::T = 0.,
    con::C = (result,x,u,ξ)->nothing,
    gl = T[],
    gu = T[],
    nc = length(gl)
    ) where {T, V <: AbstractVector{T}, R, D, P, C}

    M = length(ξs)
    
    PolicyNLPModel{T, V, R, D, P, C}(
        N,
        nc,
        x0s,
        ξs,
        γ,
        λ,
        rew,
        dyn,
        pol,
        con,
        V(undef,nc*N*M),
        NLPModelMeta{T, Vector{T}}(
            get_W_dim(pol);
            ncon = length(gl) * N*M,
            lcon = repeat(gl, N*M),
            ucon = repeat(gu, N*M)
        ),
        NLPModels.Counters()
    )
end

struct DensePolicy{F <: Function, M}
    f::F
    dims::Vector{Int}
    K::M
end

get_W_dim(pol) = sum((pol.dims[k]+1)*pol.dims[k+1] for k in 1:length(pol.dims)-1)

function (pol::DensePolicy{F})(W, x) where F

    @assert length(W) == get_W_dim(pol)

    offset = 0

    xk = x
    for k in 1:length(pol.dims)-1
        dim1 = pol.dims[k]
        dim2 = pol.dims[k+1]
        
        Wk = reshape(@view(W[offset + 1:offset + dim2*dim1]), dim2, dim1)
        bk = reshape(@view(W[offset + dim1*dim2 + 1 : offset + dim1*dim2 + dim2]), dim2)

        offset += dim1*dim2 + dim2

        if k == length(pol.dims)-1
            xk = Wk * xk + bk
        else
            xk = pol.f.(Wk * xk + bk)
        end
    end

    return xk 
end


function NLPModels.obj(nlp::PolicyNLPModel, W)
    result = nlp.λ * sum(W.^2)
    for (x0,ξ) in zip(nlp.x0s, nlp.ξs)
        xk = x0
        for k=1:nlp.N
            uk = nlp.pol(W,xk)
            result += nlp.γ^(k-1) * nlp.rew(xk,uk,ξ[k]) / length(nlp.ξs)
            if k != nlp.N
                xk = nlp.dyn(xk,uk,ξ[k])
            end
        end
    end
    
    return result
end

function NLPModels.cons(nlp::PolicyNLPModel, W::AbstractVector)
    result = []
    for (x0,ξ) in zip(nlp.x0s,nlp.ξs)
        xk = x0
        for k=1:nlp.N
            uk = nlp.pol(W,xk)
            append!(result, nlp.con(xk, uk, ξ[k]))
            if k != nlp.N
                xk = nlp.dyn(xk,uk,ξ[k])
            end
        end
    end
    return result
end

function lag(nlp::PolicyNLPModel, W::AbstractVector, l, obj_weight)
    cnt = 0
    result = nlp.λ * sum(W.^2) * length(nlp.ξs)
    for (x0,ξ) in zip(nlp.x0s,nlp.ξs)
        xk = x0
        for k=1:nlp.N
            uk = nlp.pol(W,xk)
            cnt += 1
            result += dot(view(l, nlp.nc*(cnt-1)+1:nlp.nc*cnt), nlp.con(xk, uk, ξ[k])) / length(nlp.ξs)
            result += obj_weight * nlp.γ^(k-1) * nlp.rew(xk,uk,ξ[k]) / length(nlp.ξs)
            if k != nlp.N
                xk = nlp.dyn(xk,uk,ξ[k])
            end
        end
    end
    return result
end


function NLPModels.cons!(nlp::AbstractNLPModel, W::AbstractVector, result::AbstractVector)
    result.= NLPModels.cons(nlp, W)
end

function NLPModels.grad!(nlp::AbstractNLPModel, W, g)
    ForwardDiff.gradient!(
        g,
        x->NLPModels.obj(nlp,x),
        W
    )
end

function MadNLP.hess_dense!(nlp::PolicyNLPModel, W, l, hess; obj_weight = 1.0)
    ForwardDiff.hessian!(
        hess,
        x -> obj_weight * NLPModels.obj(nlp,x)  + l' * NLPModels.cons(nlp,x),
        W
        )
end
function MadNLP.hess_dense!(nlp::AbstractNLPModel, W, l, hess; obj_weight = 1.0)
    ForwardDiff.hessian!(
        hess,
        x -> obj_weight * NLPModels.obj(nlp,x)  + l' * NLPModels.cons(nlp,x),
        W
    )
end


function MadNLP.jac_dense!(nlp::AbstractNLPModel, W, jac)
    ForwardDiff.jacobian!(
        jac,
        x -> NLPModels.cons(nlp, x),
        W
    )
end

struct MPCModel{T,VT <: AbstractVector{T}, R, D,  C} <: NLPModels.AbstractNLPModel{T, Vector{T}}
    N::Int
    nx::Int
    nu::Int
    nξ::Int
    x0::VT
    γ::T
    rew::R
    dyn::D
    con::C
    g::VT
    meta::AbstractNLPModelMeta{T, VT}
    counters::NLPModels.Counters
end


function NLPModels.obj(nlp::MPCModel, v)
    nx = nlp.nx
    nu = nlp.nu
    u_offset = nlp.N * nx
    result = 0.0    
    for k=1:nlp.N
        xk = @view v[nx*(k-1)+1 : nx*k]
        uk = @view v[u_offset + nu*(k-1)+1 : u_offset + nu*k]
        result += nlp.γ^(k-1) * nlp.rew(xk,uk)
    end
    
    return result
end

function NLPModels.cons(nlp::MPCModel, v::AbstractVector)
    nx = nlp.nx
    nu = nlp.nu
    
    u_offset = nlp.N * nx
    result = []
    append!(result, v[1 : nx])
    for k=1:nlp.N-1
        xk = @view v[nx*(k-1)+1 : nx*k]
        xk1 = @view v[nx*k+1 : nx*(k+1)]
        uk = @view v[u_offset + nu*(k-1)+1 : u_offset + nu*k]
        append!(result, xk1 - nlp.dyn(xk,uk))
    end
    for k=1:nlp.N
        xk = @view v[nx*(k-1)+1 : nx*k]
        uk = @view v[u_offset + nu*(k-1)+1 : u_offset + nu*k]
        append!(result, nlp.con(xk, uk))
    end
    return result
end


function MPCPolicy(mpc,xind,uind,yind,nx,nu,x0,xl,xu)
    mpc.cnt.k = 0
    if all( xl .<= x0[1:length(xl)] .<= xu)
        mpc.rhs[yind] .= x0
        mpc.x.values[xind] .= x0
        solve!(mpc)
        mpc.status == MadNLP.SOLVE_SUCCEEDED ? print("/") : print("*") 
        return mpc.status == MadNLP.SOLVE_SUCCEEDED ? mpc.x.values[uind] : min.(max.(K*x0,ul),uu)
    else
        print("!")
        return min.(max.(K*x0,ul),uu)
    end
end

function performance(
    rew,
    dyn,
    pol,
    con,
    gl,
    gu,
    x0s,
    ξs,
    γ,
    nx,
    nu,
    Tsim,
    Nsam
    )
    
    xs = randn(nx,Tsim)
    us = randn(nu,Tsim)
    rews = zeros(Nsam)
    cvio = zeros(Int,Nsam)

    for k=1:Nsam
        xs[:,1] = x0s[k]
        us[:,1] = pol(xs[:,1])
        
        for i=1:Tsim-1
            xs[:,i+1] = dyn(xs[:,i], us[:,i], ξs[k][:,i])
            us[:,i+1] = pol(xs[:,i+1])
            rews[k] += γ^(i-1) * rew(xs[:,i], us[:,i], ξs[k][:,i])
            cvio[k] += !all(gl .<= con(xs[:,i], us[:,i], ξs[k][:,i]) .<= gu)
        end
        rews[k] += γ^Tsim * rew(xs[:,Tsim], us[:,Tsim], ξs[k][:,Tsim])
        cvio[k] += !all(gl .<= con(xs[:,Tsim], us[:,Tsim], ξs[k][:,Tsim]) .<= gu )
    end

    return mean(rews), mean(cvio)
end
