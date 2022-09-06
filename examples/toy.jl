### A definition
A11 = [
    .9 -.05 
    -.05 .9 
]
A12 = [
    .8 0
    .8 0
]
A22 = exp([0 1; -1 0]*pi/24/60)

### B definition
B1 = [
    1 0 -1 0
    0 1 0 -1
]


### Q definition
Q1 = [
    1 0 
    0 1 
]


### R definition
R = [
    1 0 0 0 
    0 1 0 0 
    0 0 2 0 
    0 0 0 2 
]

### C definition
C1 = [
    0 0.1
    0 0.1
]

### D definition
D =  [
    0 0
    0 0
    0 0
    0 0
    0 0 
    0 0 
]

### E definition
E1 = [
    1 0 
    0 1 
    0 0 
    0 0 
    0 0 
    0 0 
]


### F definition
F = [
    0 0 0 0
    0 0 0 0
    1 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 0 1
]

S  = [
    0 0 .2 0
    0 0 .2 0
    0 0 .1 0
    0 0 .1 0
]

T  = [
    .1 0 
    .1 0 
    .1 0
    .1 0
]

gl = [
    -2.
    -2
    0
    0
    0
    0
]
gu = [
    2.
    2
    1
    1
    1
    1
]

E2 = zeros(size(E1,1),size(A22,1))
x0l = -1 * ones(4)
x0u = 1 * ones(4)
ξl = -1*ones(2)
ξu = 1*ones(2)
