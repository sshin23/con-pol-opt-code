using CUDA, ForwardDiff, Zygote, LinearAlgebra, NLPModels, MadNLP, MadNLPGPU, Random

Random.seed!(0)

struct PolicyNLPModel{R,D,P,T,V <: AbstractVector{T}} <: AbstractNLPModel{T,Vector{T}}
    rew::R
    dyn::D
    pol::P
    x0s::Vector{V}
    xis::Vector{Vector{V}}
    meta::AbstractNLPModelMeta{T,Vector{T}}
    counters::NLPModels.Counters
end

function PolicyNLPModel(
    rew::R,
    dyn::D,
    pol::P,
    x0s::Vector{V},
    xis::Vector{Vector{V}}
    ) where {R, D, P, T, V <: AbstractVector{T}}
    
    PolicyNLPModel{R, D, P, T, V}(
        rew,
        dyn,
        pol,
        x0s,
        xis,
        NLPModelMeta{T,Vector{T}}(
            get_W_dim(pol)
        ),
        NLPModels.Counters()
    )
end

struct DensePolicy{F <: Function}
    f::F
    dims::Vector{Int}
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


function NLPModels.obj(nlp::PolicyNLPModel,W)
    result = 0
    W = CuArray(W)
    
    for (x0,xi) in zip(nlp.x0s,nlp.xis)
        xk = x0
        for k=1:N
            uk = nlp.pol(W,xk)
            result += nlp.rew(xk,uk,xi[k])
            if k != N
                xk = nlp.dyn(xk,uk,xi[k])
            end
        end
    end
    
    return result
end

function NLPModels.grad!(nlp::PolicyNLPModel,W, g)
    W = CuArray(W)
    copyto!(
        g,
        Zygote.gradient(x->NLPModels.obj(nlp,x), W)[1]
    )
end

function MadNLP.hess_dense!(nlp::PolicyNLPModel,W, l, hess; obj_weight = 1.0)
    W = CuArray(W)
    copyto!(
        hess,
        ForwardDiff.jacobian(
            x->Zygote.gradient(
                x->obj_weight * NLPModels.obj(nlp,x),
                x
            )[1],
            W
        )
    )
end

function MadNLP.jac_dense!(nlp::PolicyNLPModel,W, jac) end
