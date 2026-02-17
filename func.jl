using LinearAlgebra, SparseArrays

function gaussSeidel(A, b, x; tol=1e-5, maxiter=10000)
    A = float.(A); b = float.(b); x = float.(x)
    n = size(A, 1)
    for k in 1:maxiter
        x_old = copy(x)
        for i in 1:n
            sigma = 0.0
            for j in 1:n
                j != i && (sigma += A[i,j] * x[j])
            end
            x[i] = (b[i] - sigma) / A[i,i]
        end
        norm(x - x_old, Inf) < tol && return x
    end
    error("Pas de convergence après $maxiter itérations")
end

# Laplacien 2D correct : (nx*ny) × (nx*ny) avec stencil 5 points
function laplacian2D(nx, ny)
    N  = nx * ny
    A  = spzeros(N, N)
    id(i,j) = (j-1)*nx + i
    for j in 1:ny, i in 1:nx
        k = id(i,j)
        A[k,k] = -4.0
        i > 1  && (A[k, id(i-1,j)] = 1.0)
        i < nx && (A[k, id(i+1,j)] = 1.0)
        j > 1  && (A[k, id(i,j-1)] = 1.0)
        j < ny && (A[k, id(i,j+1)] = 1.0)
    end
    return A
end

# -------------------------------------------------------
# Conditions aux bords pour grille staggered
#
#  u : (nx+1, ny)   — composante x sur faces verticales
#     indices i=1 (mur gauche) et i=nx+1 (mur droit)
#     indice  j=1..ny (intérieur vertical)
#     Le lid (j=ny) impose u=1 sur toutes les faces hautes
#
#  v : (nx, ny+1)   — composante y sur faces horizontales
#     indices j=1 (bas) et j=ny+1 (haut)
#     indice  i=1..nx
# -------------------------------------------------------
function condition_bord!(u, v)
    # u : murs gauche et droit → u=0
    u[1,   :] .= 0.0
    u[end, :] .= 0.0
    # u : mur bas → u=0
    u[:, 1]   .= 0.0
    # u : lid (haut) → u=1  (couvercle glissant)
    u[:, end] .= 10.0

    # v : mur bas et haut → v=0
    v[:, 1]   .= 0.0
    v[:, end] .= 0.0
    # v : murs gauche et droit → v=0
    v[1,   :] .= 0.0
    v[end, :] .= 0.0
end

# -------------------------------------------------------
# Divergence discrète sur grille staggered (flux nets)
# -------------------------------------------------------
function divergence(u, v, i, j, dx, dy)
    (u[i+1,j] - u[i,j]) / dx + (v[i,j+1] - v[i,j]) / dy
end

# -------------------------------------------------------
# Diffusion visqueuse  ν·∇²φ  (Laplacien centré)
# -------------------------------------------------------
function flux_diff(nu, phi, dx, dy)
    nx, ny = size(phi)
    F = zeros(nx, ny)
    for i in 2:nx-1, j in 2:ny-1
        F[i,j] = nu * (
            (phi[i+1,j] + phi[i-1,j] - 2phi[i,j]) / dx^2 +
            (phi[i,j+1] + phi[i,j-1] - 2phi[i,j]) / dy^2
        )
    end
    return F
end

# -------------------------------------------------------
# Convection  (u·∇)u  et  (u·∇)v  — schéma centré
# -------------------------------------------------------
function flux_conv_u(u, v, dx, dy)
    # u : (nx+1, ny),  v : (nx, ny+1)
    nx1, ny = size(u)
    F = zeros(nx1, ny)
    for i in 2:nx1-1, j in 2:ny-1
        # u à la face u[i,j] : advection en x et en y
        u_here = u[i,j]
        dudx = (u[i+1,j] - u[i-1,j]) / (2dx)
        # v interpolé au nœud u[i,j]  (moyenne de 4 voisins)
        v_at_u = 0.25*(v[i-1,j] + v[i,j] + v[i-1,j+1] + v[i,j+1])
        dudy = (u[i,j+1] - u[i,j-1]) / (2dy)
        F[i,j] = u_here*dudx + v_at_u*dudy
    end
    return F
end

function flux_conv_v(u, v, dx, dy)
    # v : (nx, ny+1),  u : (nx+1, ny)
    nx, ny1 = size(v)
    F = zeros(nx, ny1)
    for i in 2:nx-1, j in 2:ny1-1
        v_here = v[i,j]
        dvdy = (v[i,j+1] - v[i,j-1]) / (2dy)
        # u interpolé au nœud v[i,j]
        u_at_v = 0.25*(u[i,j-1] + u[i+1,j-1] + u[i,j] + u[i+1,j])
        dvdx = (v[i+1,j] - v[i-1,j]) / (2dx)
        F[i,j] = u_at_v*dvdx + v_here*dvdy
    end
    return F
end

# -------------------------------------------------------
# Gradient de pression pour la correction de vitesse
# -------------------------------------------------------
function gradp_u(p, dx)
    nx, ny = size(p)
    g = zeros(nx+1, ny)
    for i in 2:nx, j in 1:ny
        g[i,j] = (p[i,j] - p[i-1,j]) / dx
    end
    return g
end

function gradp_v(p, dy)
    nx, ny = size(p)
    g = zeros(nx, ny+1)
    for i in 1:nx, j in 2:ny
        g[i,j] = (p[i,j] - p[i,j-1]) / dy
    end
    return g
end