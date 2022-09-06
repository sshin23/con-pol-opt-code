using PowerModels

data = parse_file("/home/sshin/git/pglib-opf/pglib_opf_case3_lmbd.m")

n  = length(data["bus"])
L  = Array(imag.(PowerModels.calc_admittance_matrix(data).matrix))

iG = [g["gen_bus"] for (i,g) in data["gen"]]
iS = setdiff(1:n,iG)
iT = [br["t_bus"] for (i,br) in data["branch"]]
iF = [br["f_bus"] for (i,br) in data["branch"]]

nS = length(iS)
nG = length(iG)
    
# x: [x1; x2], where x1 is the current storge, and x2 is the current generation level
# u: [u1; u2], where u1 is charge/discharge rate, and u2 is ramping for dispatchable generation
# ξ: [ξ1; ξ2], where ξ1 is rewnewable generation, and ξ2 is demand
# IS: incidence matrix for storages
# IG: incidence matrix for generators
# IB: incidence matrix for branches

IS  = I[1:n,iS]
IG  = I[1:n,iG]
IB  = I[iT,1:n] - I[iF,1:n]  

A11 = I[1:n,1:n]

A12 = zeros(n,2)
A22 = exp(
    [
        0 1
        -1 0
    ] * pi/24 # daily
)
LP = zeros(n)
for (i,ld) in data["load"]
    LP[ld["load_bus"]] = ld["pd"]
end


B1  = I[1:n,1:n]

C1  = zeros(n,2)

D   = [
    zeros(2n,2)
    IB * inv(L) * LP zeros(n)
]



E1  = [
    I
    0*I
    zeros(size(IB,1),size(IS,2)) IB * inv(L) * IG;
]
E2  = [
    zeros(2n,size(D0,2))
    IB * inv(L) * [
        LP zeros(n)
    ]
]
    
F   = [
    zeros(n,size(B1,2))
    I
    IB * inv(L) * IS zeros(size(IB,1),size(IG,2));
]
    

Q1  = I[1:n,1:n]
R   = I[1:n,1:n]
S   = 0*I[1:n,1:n+2]
T   = zeros(n,2)


gl  = [
    0.0 * ones(n)
    -0.001 * ones(n)
    - pi/6 * ones(size(IB,1))
]
gu  = [
    1 * ones(n)
    0.001 * ones(n)
    pi/6 * ones(size(IB,1))
]

x0l = [
    .5 * ones(n);
    -ones(2)
]
x0u = [
    .5 * ones(n)
    ones(2)
]

ξl = -1 * ones(2)
ξu = 1 * ones(2)
