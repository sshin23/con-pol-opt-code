Random.seed!(0)

nx = 4
nu = 2
M  = 10
N  = 100

A = CUDA.randn(nx,nx) 
B = CUDA.randn(nx,nu)

rew(x,u,xi) = (1/2)*dot(x,x) + (1/2)*dot(u,u)
dyn(x,u,xi) = A*x + B*u + xi
pol = DensePolicy(tanh, [nx,nu])

W = CUDA.randn(get_W_dim(pol)) * 0.001
x0s= [CUDA.randn(nx) for i=1:M]
xis= [[CUDA.randn(nx) for k=1:N] for i=1:M]

noise = CUDA.randn(get_W_dim(pol)) * 0.1

nlp = PolicyNLPModel(rew,dyn,pol,x0s,xis)

Acpu = Array(A); Bcpu= Array(B); P = ared(Acpu,Bcpu,I,I)[1]; K=-inv(I+Bcpu'*P*Bcpu)*Bcpu'*P*Acpu;
W_init = CuArray([K[:];zeros(nu)]) + noise
copyto!(nlp.meta.x0, W_init)

solver = MadNLPGPU.CuInteriorPointSolver(
    nlp;
    kkt_system=MadNLP.DENSE_KKT_SYSTEM,
    tol = 1e-3,
    lapack_algorithm=MadNLP.CHOLESKY
)

MadNLP.optimize!(solver)


copyto!(W, W_init)
eta = 0.000001

for i=1:1000
    W .-= eta * Zygote.gradient(x->NLPModels.obj(nlp,x), W)[1]
    println(obj(nlp,W))
end



W = CuArray(solver.x)
T = 10000
xs = CUDA.randn(nx,T)
for i=1:T-1
    xs[:,i+1] = A*xs[:,i] + B*pol(W,xs[:,i])
end
