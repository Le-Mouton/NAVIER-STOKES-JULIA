using LinearAlgebra, SparseArrays, Plots
include("func.jl")




function main()
    nx, ny = 64, 64
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx/nx, Ly/ny

    rho = 1.2
    nu  = 0.16
    dt  = 0.2 * dx^2 / nu
    nt  = 5000

    # Fréquence d'enregistrement des frames (40 frames au total)
    n_frames   = 40
    frame_step = max(1, nt ÷ n_frames)

    u = zeros(nx+1, ny)
    v = zeros(nx, ny+1)
    p = zeros(nx, ny)

    A_p = laplacian2D(nx, ny) / dx^2
    A_p[1,:] .= 0.0
    A_p[1,1]  = 1.0

    xs = ((1:nx) .- 0.5) .* dx
    ys = ((1:ny) .- 0.5) .* dy

    println("dt = $(round(dt, sigdigits=3)),  nt = $nt,  frame tous les $frame_step pas")

    anim = @animate for n in 1:nt
        condition_bord!(u, v)

        Du = flux_diff(nu, u, dx, dy)
        Dv = flux_diff(nu, v, dx, dy)
        Cu = flux_conv_u(u, v, dx, dy)
        Cv = flux_conv_v(u, v, dx, dy)

        u_star = u .+ dt .* (Du .- Cu)
        v_star = v .+ dt .* (Dv .- Cv)
        condition_bord!(u_star, v_star)

        b = zeros(nx * ny)
        for j in 2:ny-1, i in 2:nx-1
            k = (j-1)*nx + i
            b[k] = (rho/dt) * divergence(u_star, v_star, i, j, dx, dy)
        end
        b[1] = 0.0

        p_vec = A_p \ b
        p .= reshape(p_vec, nx, ny)

        u .= u_star .- (dt/rho) .* gradp_u(p, dx)
        v .= v_star .- (dt/rho) .* gradp_v(p, dy)
        condition_bord!(u, v)

        # Recentrer aux cellules pour le plot
        uc = 0.5 .* (u[1:end-1,:] .+ u[2:end,:])
        vc = 0.5 .* (v[:,1:end-1] .+ v[:,2:end])

        gif_maker(uc, vc, p, xs, ys, dx, dy, n, nt)


    end every frame_step

    gif(anim, "navier_stokes.gif", fps=15)
    println("GIF sauvegardé.")

    return u, v, p, dx, dy, nx, ny
end

u, v, p, dx, dy, nx, ny = main()

# Recentrer u et v aux centres des cellules
uc = 0.5 .* (u[1:end-1,:] .+ u[2:end,:])
vc = 0.5 .* (v[:,1:end-1] .+ v[:,2:end])
speed = sqrt.(uc.^2 .+ vc.^2)

# Grille centres de cellules
xs = ((1:nx) .- 0.5) .* dx
ys = ((1:ny) .- 0.5) .* dy

p1 = heatmap(xs, ys, uc',
    title="u (vitesse horizontale)", xlabel="x", ylabel="y",
    aspect_ratio=1, color=:RdBu)

p2 = heatmap(xs, ys, vc',
    title="v (vitesse verticale)", xlabel="x", ylabel="y",
    aspect_ratio=1, color=:RdBu)

p3 = heatmap(xs, ys, p',
    title="p (pression)", xlabel="x", ylabel="y",
    aspect_ratio=1, color=:viridis)

# Champ vectoriel
step = 2
xi = 1:step:nx
yj = 1:step:ny
Xg = [xs[i] for i in xi, j in yj]
Yg = [ys[j] for i in xi, j in yj]
sp = speed[xi, yj] .+ 1e-10
Un = uc[xi, yj] ./ sp .* dx .* step .* 0.5
Vn = vc[xi, yj] ./ sp .* dy .* step .* 0.5

p4 = heatmap(xs, ys, speed',
    title="Champ de vitesse (norme)", xlabel="x", ylabel="y",
    aspect_ratio=1, color=:RdBu)
quiver!(p4, vec(Xg), vec(Yg), quiver=(vec(Un), vec(Vn)),
    arrow=true, color=:white, linewidth=0.8)

plot(p1, p2, p3, p4, layout=(2,2), size=(1000,900))
savefig("navier_stokes.png")