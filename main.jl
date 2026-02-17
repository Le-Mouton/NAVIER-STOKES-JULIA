using LinearAlgebra
using SparseArrays
using Plots

function gaussSeidel(A, b, x; tol=1e-5, maxiter=10000)
    A = float.(A)
    b = float.(b)
    x = float.(x)
    n = size(A,1)

    for k in 1:maxiter
        x_old = copy(x)

        for i in 1:n
            σ = 0.0
            for j in 1:n
                if j != i
                    σ += A[i,j] * x[j]
                end
            end
            x[i] = (b[i] - σ) / A[i,i]
        end

        if norm(x - x_old, Inf) < tol
            return x
        end
    end
    error("Pas de convergence")
end

function laplacian(A, i, j, dx, dy)
    (A[i+1,j] - 2A[i,j] + A[i-1,j])/(dx^2) + (A[i,j+1] - 2A[i,j] + A[i,j-1])/(dy^2)
end

function laplacian2D(nx, ny, dx, dy)
    N = nx*ny
    A = spzeros(Float64, N, N)

    ax = 1/dx^2
    ay = 1/dy^2
    c  = -2ax - 2ay

    idx(i,j) = i + (j-1)*nx

    for j in 1:ny, i in 1:nx
        k = idx(i,j)

        if i==1 || i==nx || j==1 || j==ny
            A[k,k] = 1.0
        else
            A[k,k] = c
            A[k, idx(i-1,j)] = ax
            A[k, idx(i+1,j)] = ax
            A[k, idx(i,j-1)] = ay
            A[k, idx(i,j+1)] = ay
        end
    end

    return A
end

function ddx(A, i, j, dx)
    (A[i+1,j] - A[i-1,j])/(2dx)
end

function ddy(A, i, j, dy)
    (A[i,j+1] - A[i,j-1])/(2dy)
end

function divergence(u, v, i, j, dx, dy)
    ddx(u, i, j, dx) + ddy(v, i, j, dy)
end

function maillage(u, v, p)
    nx, ny = size(u)

    # vitesses: bords à 0 (no-slip partout)
    u[1,:] .= 0; u[end,:] .= 0; u[:,1] .= 0; u[:,end] .= 0
    v[1,:] .= 0; v[end,:] .= 0; v[:,1] .= 0; v[:,end] .= 0

    # pression: Neumann dp/dn=0
    p[1,:]   .= p[2,:]
    p[end,:] .= p[end-1,:]
    p[:,1]   .= p[:,2]
    p[:,end] .= p[:,end-1]
end

# step "évolutif" : on passe ustar/vstar et A, et on modifie u/v/p en place
function step(u, v, ustar, vstar, p, A, rho, nu, dt, dx, dy)
    nx, ny = size(u)

    ustar .= u
    vstar .= v

    # Prediction
    for i in 2:nx-1, j in 2:ny-1
        adv_u = u[i,j]*ddx(u,i,j,dx) + v[i,j]*ddy(u,i,j,dy)
        adv_v = u[i,j]*ddx(v,i,j,dx) + v[i,j]*ddy(v,i,j,dy)

        ustar[i,j] = u[i,j] + dt * (-adv_u + nu*laplacian(u,i,j,dx,dy))
        vstar[i,j] = v[i,j] + dt * (-adv_v + nu*laplacian(v,i,j,dx,dy))
    end

    maillage(ustar, vstar, p)

    rhs = zeros(nx, ny)
    for i in 2:nx-1, j in 2:ny-1
        rhs[i,j] = (rho/dt) * divergence(ustar, vstar, i, j, dx, dy)
    end

    # Résoudre Poisson via ton GS matriciel
    p_vec = gaussSeidel(A, vec(rhs), vec(p); tol=1e-6, maxiter=500)
    p .= reshape(p_vec, nx, ny)

    maillage(ustar, vstar, p)

    # Correction
    for i in 2:nx-1, j in 2:ny-1
        u[i,j] = ustar[i,j] - (dt/rho) * ddx(p,i,j,dx)
        v[i,j] = vstar[i,j] - (dt/rho) * ddy(p,i,j,dy)
    end

    maillage(u, v, p)
end

nx, ny = 64, 64
Lx, Ly = .5, .5
dx, dy = Lx/(nx-1), Ly/(ny-1)

rho = 1.0
dt = 0.01

ustar = zeros(nx, ny)
vstar = zeros(nx, ny)

for j in 1:ny, i in 1:nx
    ustar[i,j] = sin(pi*i/nx)
end

rhs = divergence_rhs(ustar, vstar, rho, dt, dx, dy)

A = laplacian2D(nx, ny, dx, dy)
b = vec(rhs)
x0 = zeros(nx*ny)

p_vec = gaussSeidel(A, b, x0; tol=1e-5, maxiter=2000)
p = reshape(p_vec, nx, ny)
println(p[16,16])

x = range(0, Lx, length=nx)
y = range(0, Ly, length=ny)

cont = contourf(x, y, p', levels=20, xlabel="x", ylabel="y", c=:viridis)
savefig(cont, "contourf.png")