using LinearAlgebra

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


function laplacian2D(nx, ny, dx, dy; anchor=(1,1))
    N = nx*ny
    A = spzeros(Float64, N, N)

    ax = 1/dx^2
    ay = 1/dy^2

    idx(i,j,nx) = i + (j-1)*nx

    for j in 1:ny, i in 1:nx
        k = idx(i,j,nx)

        diag = 0.0

        # x-
        if i > 1
            A[k, idx(i-1,j,nx)] = ax
            diag += ax
        else
            # Neumann: p(0,j)=p(1,j) -> terme x- contribue 0, donc rien
        end

        # x+
        if i < nx
            A[k, idx(i+1,j,nx)] = ax
            diag += ax
        else
        end

        # y-
        if j > 1
            A[k, idx(i,j-1,nx)] = ay
            diag += ay
        end

        # y+
        if j < ny
            A[k, idx(i,j+1,nx)] = ay
            diag += ay
        end

        A[k,k] = -diag
    end

    # ancrage (pression définie à une constante près en Neumann pur)
    ia, ja = anchor
    ka = idx(ia, ja, nx)
    A[ka, :] .= 0.0
    A[ka, ka] = 1.0

    return A
end

function condition_bord(u, v, p)

    u[1,:] .= 0 # mur de gauche
    v[1,:] .= 0 

    u[:,1] .= 0 # mur bas
    v[:,1] .= 0

    u[end,:] .= 0 # droite
    v[end,:] .= 0

    u[:,end] .= 0.01 # en haut
    v[:,end] .= 0
end

function divergence(u, v, i, j, dx, dy)
    ddx(u, i, j, dx) + ddy(v, i, j, dy)
end

function interpolate_x(A, i, j)
    (A[i+1,j] + A[i,j])/(2)
end

function interpolate_y(A, i, j)
    (A[i,j+1] + A[i,j])/(2)
end

function ddx(A, i, j, dx)
    (A[i+1,j] - A[i,j])/(2*dx)
end

function ddy(A, i, j, dy)
    (A[i,j+1] - A[i,j])/(2*dy)
end

function dd2x(A, i, j, dx)
    (A[i+1,j] + A[i-1,j] - 2*A[i, j])/(dx^2)
end

function dd2y(A, i, j, dy)
    (A[i,j+1] + A[i,j-1] - 2*A[i, j])/(dy^2)
end

function flux_diff(nu, A, dx, dy)
    nx, ny = size(A)

    flux_x = zeros(nx, ny)
    flux_y = zeros(nx, ny)

    for i in 2:nx-1, j in 2:ny-1
        flux_x[i, j] = nu*dy*ddx(A, i, j, dx)
    end
    for i in 2:nx-1, j in 2:ny-1
        flux_y[i, j] = nu*dx*ddy(A, i, j, dy)
    end

    return flux_x + flux_y
end

function flux_conv(u, v, dx)
    nx_u, ny_u = size(u)
    nx_v, ny_v = size(v)

    flux_x = zeros(nx_u, ny_u)
    flux_y = zeros(nx_u, ny_u)

    for i in 2:nx_u-1, j in 2:ny_u-1
        flux_x[i, j] = interpolate_x(u, i, j) * u[i, j] * dx
    end

    i_max = min(nx_u-1, nx_v)
    j_max = min(ny_u-1, ny_v-1)

    for i in 2:i_max, j in 2:j_max
        flux_y[i, j] = interpolate_y(v, i, j) * u[i, j] * dx
    end

    return flux_x + flux_y
end

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