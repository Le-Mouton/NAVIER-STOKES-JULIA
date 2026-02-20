using LinearAlgebra, SparseArrays, Plots

function gaussEidel(At::SparseMatrixCSC{T,Int}, b::AbstractVector, x::AbstractVector;
                    tol=1e-6, maxiter=20_000, ω=1.6) where {T<:Real}
    rows = rowvals(At); vals = nonzeros(At)
    n = size(At, 1)
    @assert size(At,2) == n  "At doit être carrée"
    @assert length(b)  == n  "b mauvaise taille"
    @assert length(x)  == n  "x mauvaise taille"
    @assert 0.0 < ω < 2.0   "ω doit être dans (0,2) pour SOR"

    for k in 1:maxiter
        δmax = zero(T)
        for i in 1:n
            sigma = zero(T)
            aii   = zero(T)
            found_diag = false
            for ptr in nzrange(At, i)
                j = rows[ptr]; aij = vals[ptr]
                if j == i
                    aii = aij; found_diag = true
                else
                    sigma += aij * x[j]
                end
            end
            !found_diag || aii == 0 && error("SOR: diagonale nulle/manquante à i=$i")
            x_new  = (1-ω)*x[i] + ω*(b[i] - sigma)/aii
            δmax   = max(δmax, abs(x_new - x[i]))   # critère sans allocation
            x[i]   = x_new
        end
        δmax < tol && return x, k
    end

    @warn "Pas de convergence après $maxiter itérations (ω=$ω)"
    return x, maxiter
end

function laplacian2D(nx, ny, dx, dy)
    N = nx * ny
    A = spzeros(N, N)
    id(i, j) = (j - 1) * nx + i

    for j in 1:ny, i in 1:nx
        k = id(i, j)

        if i == 1 && j == 1
            A[k, k] = 1.0
            continue
        end

        diag = 0.0

        if i > 1
            A[k, id(i-1, j)] += 1.0 / dx^2
            diag -= 1.0 / dx^2
        end

        if i < nx
            A[k, id(i+1, j)] += 1.0 / dx^2
            diag -= 1.0 / dx^2
        end

        if j > 1
            A[k, id(i, j-1)] += 1.0 / dy^2
            diag -= 1.0 / dy^2
        end

        if j < ny
            A[k, id(i, j+1)] += 1.0 / dy^2
            diag -= 1.0 / dy^2
        end

        A[k, k] = diag
    end
    return A
end


function condition_bord!(u, v, U0=5.0)
    # Tout à zéro
    u[1,:]   .= 0.0; u[end,:] .= 0.0
    u[:,1]   .= U0; u[:,end] .= 0.0
    v[:,1]   .= 0.0; v[:,end] .= 0.0
    v[1,:]   .= 0.0; v[end,:] .= 0.0
end

function divergence(u, v, i, j, dx, dy)
    (u[i+1, j] - u[i, j]) / dx + (v[i, j+1] - v[i, j]) / dy
end

function flux_diff(nu, phi, dx, dy)
    nx, ny = size(phi)
    F = zeros(nx, ny)
    for j in 2:ny-1, i in 2:nx-1
        F[i, j] = nu * (
            (phi[i+1, j] - 2*phi[i, j] + phi[i-1, j]) / dx^2 +
            (phi[i, j+1] - 2*phi[i, j] + phi[i, j-1]) / dy^2
        )
    end
    return F
end

function flux_conv_u(u, v, dx, dy)
    nx1, ny = size(u)
    F = zeros(nx1, ny)

    for j in 2:ny-1, i in 2:nx1-1

        u_right = i < nx1   ? (u[i, j] + u[i+1, j]) / 2 : 0.0
        u_left  = i > 1     ? (u[i-1, j] + u[i, j]) / 2 : 0.0

        il = max(i-1, 1); ir = min(i, nx1-1)
        v_top = (v[il, j+1] + v[ir, j+1]) / 2
        v_bot = (v[il, j]   + v[ir, j])   / 2

        u_adv_right = u_right >= 0 ? u[i, j]     : (i < nx1 ? u[i+1, j] : u[i, j])
        u_adv_left  = u_left  >= 0 ? (i > 1 ? u[i-1, j] : u[i, j]) : u[i, j]
        u_adv_top   = v_top   >= 0 ? u[i, j]     : (j < ny ? u[i, j+1] : u[i, j])
        u_adv_bot   = v_bot   >= 0 ? (j > 1 ? u[i, j-1] : u[i, j]) : u[i, j]

        F[i, j] = (u_right * u_adv_right - u_left * u_adv_left) / dx +
                  (v_top   * u_adv_top   - v_bot  * u_adv_bot)  / dy
    end
    return F
end

function flux_conv_v(u, v, dx, dy)
    nx, ny1 = size(v)
    F = zeros(nx, ny1)

    for j in 2:ny1-1, i in 2:nx-1
        v_top   = j < ny1   ? (v[i, j] + v[i, j+1]) / 2 : 0.0
        v_bot   = j > 1     ? (v[i, j-1] + v[i, j]) / 2 : 0.0

        jb = max(j-1, 1); jt = min(j, ny1-1)
        u_right = (u[i+1, jb] + u[i+1, jt]) / 2
        u_left  = (u[i,   jb] + u[i,   jt]) / 2

        v_adv_top   = v_top   >= 0 ? v[i, j]     : (j < ny1 ? v[i, j+1] : v[i, j])
        v_adv_bot   = v_bot   >= 0 ? (j > 1 ? v[i, j-1] : v[i, j]) : v[i, j]
        v_adv_right = u_right >= 0 ? v[i, j]     : (i < nx ? v[i+1, j] : v[i, j])
        v_adv_left  = u_left  >= 0 ? (i > 1 ? v[i-1, j] : v[i, j]) : v[i, j]

        F[i, j] = (u_right * v_adv_right - u_left * v_adv_left) / dx +
                  (v_top   * v_adv_top   - v_bot  * v_adv_bot)  / dy
    end
    return F
end

function gradp_u(p, dx)
    nx, ny = size(p)
    g = zeros(nx+1, ny)
    for j in 1:ny, i in 2:nx
        g[i, j] = (p[i, j] - p[i-1, j]) / dx
    end
    return g
end

function gradp_v(p, dy)
    nx, ny = size(p)
    g = zeros(nx, ny+1)
    for j in 2:ny, i in 1:nx
        g[i, j] = (p[i, j] - p[i, j-1]) / dy
    end
    return g
end

function gif_maker(uc, vc, p, xs, ys, dx, dy, n, nt)
    nx, ny = size(uc)
    speed  = sqrt.(uc.^2 .+ vc.^2)

    step = max(1, nx ÷ 20)
    xi = 1:step:nx
    yj = 1:step:ny
    Xg = [xs[i] for i in xi, j in yj]
    Yg = [ys[j] for i in xi, j in yj]
    sp = speed[xi, yj] .+ 1e-10
    Un = uc[xi, yj] ./ sp .* dx .* step .* 0.5
    Vn = vc[xi, yj] ./ sp .* dy .* step .* 0.5

    p1 = heatmap(xs, ys, uc',
        title="u (vitesse horizontale)", xlabel="x", ylabel="y",
        aspect_ratio=1, color=:RdBu, clims=(-1, 1))

    p2 = heatmap(xs, ys, vc',
        title="v (vitesse verticale)", xlabel="x", ylabel="y",
        aspect_ratio=1, color=:RdBu, clims=(-1, 1))

    p3 = heatmap(xs, ys, p',
        title="p (pression)", xlabel="x", ylabel="y",
        aspect_ratio=1, color=:viridis)

    pquiver = heatmap(xs, ys, speed',
        color=:plasma, clims=(0, 1),
        aspect_ratio=1,
        title="‖u‖  —  t=$(n)",
        xlabel="x", ylabel="y",
        size=(600, 550))

    return plot(p1, p2, p3, pquiver, layout=(2, 2), size=(1000, 900))
end

function streamlines(uc, vc, xs, ys; nseeds=20, nsteps=2000, ds=4e-4)
    nx, ny   = length(xs), length(ys)
    x1, y1   = xs[1], ys[1]
    ddx, ddy = xs[2]-xs[1], ys[2]-ys[1]
    xmin, xmax = xs[1], xs[end]
    ymin, ymax = ys[1], ys[end]

    @inline function interp2(f, x, y)
        xi = clamp((x-x1)/ddx + 1.0, 1.0, nx-1e-8)
        yi = clamp((y-y1)/ddy + 1.0, 1.0, ny-1e-8)
        i = floor(Int, xi); fx = xi - i
        j = floor(Int, yi); fy = yi - j
        (1-fx)*(1-fy)*f[i,j] + fx*(1-fy)*f[i+1,j] +
        (1-fx)*fy*f[i,j+1]   + fx*fy*f[i+1,j+1]
    end

    lines = Vector{Tuple{Vector{Float32}, Vector{Float32}}}()
    for si in LinRange(xmin+ddx, xmax-ddx, nseeds)
        for sj in LinRange(ymin+ddy, ymax-ddy, nseeds÷2)
            px = Float32[si]; py = Float32[sj]
            x, y = Float64(si), Float64(sj)
            for _ in 1:nsteps
                u1 = interp2(uc, x, y); v1 = interp2(vc, x, y)
                sp1 = sqrt(u1^2 + v1^2) + 1e-12
                x2 = clamp(x + ds*u1/sp1, xmin, xmax)
                y2 = clamp(y + ds*v1/sp1, ymin, ymax)
                u2 = interp2(uc, x2, y2); v2 = interp2(vc, x2, y2)
                um = (u1+u2)*0.5; vm = (v1+v2)*0.5
                sp = sqrt(um^2 + vm^2) + 1e-12
                x = clamp(x + ds*um/sp, xmin, xmax)
                y = clamp(y + ds*vm/sp, ymin, ymax)
                push!(px, x); push!(py, y)
            end
            push!(lines, (px, py))
        end
    end
    return lines
end

function plot_streamlines(uc, vc, xs, ys; nseeds=20, clim=nothing)
    speed = sqrt.(uc.^2 .+ vc.^2)
    cmax  = isnothing(clim) ? maximum(speed) : clim
    lines = streamlines(uc, vc, xs, ys; nseeds=nseeds)

    plt = heatmap(xs, ys, speed';
        color=:inferno, clims=(0.0, cmax),
        aspect_ratio=1, xlabel="x", ylabel="y",
        title="Lignes de courant  (‖u‖ en fond)",
        size=(600, 560), dpi=100)

    for (px, py) in lines
        plot!(plt, px, py; lw=0.8, lc=:white, alpha=0.6, label=false)
    end
    return plt
end