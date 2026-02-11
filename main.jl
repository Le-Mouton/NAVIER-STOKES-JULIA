using LinearAlgebra
include("func.jl")
using SparseArrays
using Plots


function main()
    nx, ny = 32, 32
    dx, dy = 1, 1
    rho = 1.2
    dt = 0.005
    nu = 0.05
    nt = 100

    u = zeros(nx+1, ny)
    v = zeros(nx, ny+1)
    p = zeros(nx, ny)

    b = zeros(nx, ny)

    A_p = laplacian2D(nx, ny, dx, dy)

    for n in 1:nt
        Fc_u = flux_conv(u, v, dx)
        Fc_v = flux_conv(v, u, dy)

        Fd_u = flux_diff(nu, u, dx, dy)
        Fd_v = flux_diff(nu, v, dx, dy)

        u_star = u .+ dt .* (Fd_u .- Fc_u)
        v_star = v .+ dt .* (Fd_v .- Fc_v)

        condition_bord(u_star, v_star, p)

        b .= 0
        for i in 2:nx-1, j in 2:ny-1
            b[i,j] = (rho/dt) * divergence(u_star, v_star, i, j, dx, dy)
        end

        p_vec = gaussSeidel(A_p, vec(b), vec(p); tol=1e-6, maxiter=1000)
        p .= reshape(p_vec, nx, ny)

        u .= u_star .- (dt/rho) .* gradp_u(p, dx)
        v .= v_star .- (dt/rho) .* gradp_v(p, dy)
    end

    return u, v, p
end

u, v, p = main()

using Plots

nx, ny = 32, 32
# --- 1) heatmaps
p1 = heatmap(u', title="u (staggered)", aspect_ratio=1)
p2 = heatmap(v', title="v (staggered)", aspect_ratio=1)
p3 = heatmap(p', title="p (cell centers)", aspect_ratio=1)

# --- 2) recentrer u,v au centre cellule pour avoir même taille (nx, ny)
uc = 0.5 .* (u[1:end-1,:] .+ u[2:end,:])     # (nx, ny)
vc = 0.5 .* (v[:,1:end-1] .+ v[:,2:end])     # (nx, ny)

# 2) option: normaliser pour voir juste les directions (champ directionnel)
speed = sqrt.(uc.^2 .+ vc.^2) .+ 1e-12
u_dir = uc ./ speed
v_dir = vc ./ speed

# grille complète des centres de cellules
X = repeat(collect(1:nx), 1, ny)      # nx×ny
Y = repeat(collect(1:ny)', nx, 1)     # nx×ny

# champ vectoriel : 1 flèche par point
pvec = quiver(
    vec(X), vec(Y),
    quiver=(vec(u_dir), vec(v_dir)),
    aspect_ratio=1,
    title="Champ vectoriel vitesse",
    xlabel="x", ylabel="y",
    size=(750,750)
)

# --- 4) afficher tout ensemble
plot(p1, p2, p3, pvec, layout=(2,2), size=(900,900))