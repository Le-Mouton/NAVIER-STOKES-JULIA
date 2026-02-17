using LinearAlgebra, SparseArrays
include("func.jl")

function main()
    nx, ny = 64, 64
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx/nx, Ly/ny

    rho = 1.0
    nu  = 0.1          # viscosité cinématique
    # Critère CFL diffusif : dt < dx²/(4ν)
    dt  = 0.2 * dx^2 / nu
    nt  = 2000

    # Grille staggered
    #   u : composante x  → (nx+1) faces verticales × ny lignes
    #   v : composante y  → nx colonnes × (ny+1) faces horizontales
    u = zeros(nx+1, ny)
    v = zeros(nx, ny+1)
    p = zeros(nx, ny)

    # Matrice de Poisson et jauge (p[1,1]=0)
    A_p = laplacian2D(nx, ny) / dx^2   # dx=dy ici
    A_p[1,:] .= 0.0
    A_p[1,1]  = 1.0

    println("dt = $dt,  nt = $nt")

    for n in 1:nt
        condition_bord!(u, v)

        # ---- Étape 1 : vitesse intermédiaire u* ----
        Du = flux_diff(nu, u, dx, dy)
        Dv = flux_diff(nu, v, dx, dy)
        Cu = flux_conv_u(u, v, dx, dy)
        Cv = flux_conv_v(u, v, dx, dy)

        u_star = u .+ dt .* (Du .- Cu)
        v_star = v .+ dt .* (Dv .- Cv)
        condition_bord!(u_star, v_star)

        # ---- Étape 2 : RHS de Poisson ----
        b = zeros(nx * ny)
        for j in 2:ny-1, i in 2:nx-1
            k = (j-1)*nx + i
            b[k] = (rho/dt) * divergence(u_star, v_star, i, j, dx, dy)
        end
        b[1] = 0.0   # jauge

        # ---- Étape 3 : résolution Poisson ----
        p_vec = A_p \ b
        p .= reshape(p_vec, nx, ny)

        # ---- Étape 4 : correction de vitesse ----
        u .= u_star .- (dt/rho) .* gradp_u(p, dx)
        v .= v_star .- (dt/rho) .* gradp_v(p, dy)
        condition_bord!(u, v)

        if n % 200 == 0
            div_max = maximum(abs(divergence(u, v, i, j, dx, dy))
                              for i in 2:nx-1, j in 2:ny-1)
            println("n=$n  |div|_max = $(round(div_max, sigdigits=3))")
        end
    end
    return u, v, p, dx, dy, nx, ny
end

u, v, p, dx, dy, nx, ny = main()

# ============================================================
# Visualisation
# ============================================================
using Plots

# Recentrer u et v aux centres des cellules
uc = 0.5 .* (u[1:end-1,:] .+ u[2:end,:])     # (nx, ny)
vc = 0.5 .* (v[:,1:end-1] .+ v[:,2:end])     # (nx, ny)
speed = sqrt.(uc.^2 .+ vc.^2)

# Grille physique (centres de cellules)
xs = ((1:nx) .- 0.5) .* dx    # vecteur nx
ys = ((1:ny) .- 0.5) .* dy    # vecteur ny

p1 = heatmap(xs, ys, uc',
    title="u (vitesse horizontale)", xlabel="x", ylabel="y",
    aspect_ratio=1, color=:RdBu)

p2 = heatmap(xs, ys, vc',
    title="v (vitesse verticale)", xlabel="x", ylabel="y",
    aspect_ratio=1, color=:RdBu)

p3 = heatmap(xs, ys, p',
    title="p (pression)", xlabel="x", ylabel="y",
    aspect_ratio=1, color=:viridis)

# Champ vectoriel (sous-échantillonné pour la lisibilité)
step = 2
xi = 1:step:nx
yj = 1:step:ny
Xg = [xs[i] for i in xi, j in yj]
Yg = [ys[j] for i in xi, j in yj]
sp = speed[xi, yj] .+ 1e-10
Un = uc[xi, yj] ./ sp .* dx .* step .* 0.8
Vn = vc[xi, yj] ./ sp .* dy .* step .* 0.8

p4 = heatmap(xs, ys, speed',
    title="Champ de vitesse (norme)", xlabel="x", ylabel="y",
    aspect_ratio=1, color=:hot)
quiver!(p4, vec(Xg), vec(Yg), quiver=(vec(Un), vec(Vn)),
    arrow=true, color=:white, linewidth=0.8)

plot(p1, p2, p3, p4, layout=(2,2), size=(1000,900))
#savefig("navier_stokes.png")
#println("Figure sauvegardée.")