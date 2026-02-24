using LinearAlgebra, SparseArrays, Plots
include("func.jl")

function main()
    nx, ny = 256, 256
    Lx, Ly = 1.0, 1.0
    dx, dy  = Lx / nx, Ly / ny
    U0      = 10.0
    rho     = 1.2
    Re      = 10000.0
    nu      = U0 * Lx / Re
    dt_diff = 0.25 * min(dx, dy)^2 / nu
    dt_conv = 0.5  * min(dx, dy) / U0
    dt      = min(dt_diff, dt_conv)
    T_final = 10.0
    nt      = ceil(Int, T_final / dt)
    n_frames   = 60
    frame_step = max(1, nt ÷ n_frames)

    println("Re=$(Re),  nu=$(round(nu,sigdigits=3))")
    println("dt=$(round(dt,sigdigits=3)),  nt=$nt,  frame tous les $frame_step pas")

    # ── Champs ────────────────────────────────────────────────────────────
    u = zeros(nx+1, ny)
    v = zeros(nx,   ny+1)
    p = zeros(nx,   ny)

    # ── Matrice Poisson ───────────────────────────────────────────────────
    A_p  = laplacian2D(nx, ny, dx, dy)
    Afix = SparseMatrixCSC(copy(A_p))
    Afix[1, :] .= 0.0
    Afix[1, 1]  = 1.0
    dropzeros!(Afix)
    At_cached = SparseMatrixCSC(transpose(Afix))   # pré-calculé UNE SEULE FOIS

    xs = ((1:nx) .- 0.5) .* dx
    ys = ((1:ny) .- 0.5) .* dy

    ω_opt = let rho_J = (cos(π/nx) + cos(π/ny)) / 2
        2 / (1 + sqrt(1 - rho_J^2))
    end
    println("ω optimal = $(round(ω_opt, sigdigits=5))")

    # ── Buffers pré-alloués — JAMAIS réaffectés dans la boucle ───────────
    Du     = zeros(nx+1, ny)
    Dv     = zeros(nx,   ny+1)
    Cu     = zeros(nx+1, ny)
    Cv     = zeros(nx,   ny+1)
    gu     = zeros(nx+1, ny)
    gv     = zeros(nx,   ny+1)
    u_star = zeros(nx+1, ny)
    v_star = zeros(nx,   ny+1)
    bvec   = zeros(nx*ny)
    pvec   = zeros(nx*ny)
    uc     = zeros(nx,   ny)
    vc     = zeros(nx,   ny)

    tol_poisson = max(1e-4, 1e-3 * nu)

    # Pré-calcul quiver (constant)
    step = max(1, nx ÷ 20)
    xi   = 1:step:nx
    yj   = 1:step:ny
    Xg   = vec([xs[i] for i in xi, j in yj])
    Yg   = vec([ys[j] for i in xi, j in yj])

    # ── Boucle principale ─────────────────────────────────────────────────
    anim = @animate for n in 1:nt
        n % 100 == 0 && print("\rÉtape $n / $nt")

        condition_bord!(u, v, U0)

        # Diffusion et convection IN-PLACE → zéro allocation
        flux_diff!(Du, nu, u, dx, dy)
        flux_diff!(Dv, nu, v, dx, dy)
        flux_conv_u!(Cu, u, v, dx, dy)
        flux_conv_v!(Cv, u, v, dx, dy)

        # Vitesse intermédiaire IN-PLACE
        @. u_star = u + dt*(Du - Cu)
        @. v_star = v + dt*(Dv - Cv)
        condition_bord!(u_star, v_star, U0)

        # RHS Poisson IN-PLACE
        fill!(bvec, 0.0)
        @inbounds for j in 1:ny, i in 1:nx
            bvec[(j-1)*nx+i] = (rho/dt) * divergence(u_star, v_star, i, j, dx, dy)
        end
        bvec[1] = 0.0

        # SOR avec warm-start IN-PLACE
        pvec .= vec(p)
        gaussEidel(At_cached, bvec, pvec; tol=tol_poisson, maxiter=5_000, ω=ω_opt)
        p .= reshape(pvec, nx, ny)

        # Correction vitesse IN-PLACE
        gradp_u!(gu, p, dx)
        gradp_v!(gv, p, dy)
        @. u = u_star - (dt/rho)*gu
        @. v = v_star - (dt/rho)*gv
        condition_bord!(u, v, U0)

        # Centrage IN-PLACE
        @inbounds for j in 1:ny, i in 1:nx
            uc[i,j] = 0.5*(u[i,j] + u[i+1,j])
            vc[i,j] = 0.5*(v[i,j] + v[i,j+1])
        end

        gif_maker(uc, vc, p, xs, ys, dx, dy, n, Xg, Yg, xi, yj, step)

    end every frame_step

    println()
    gif(anim, "navier_stokes.gif", fps=15)
    println("GIF sauvegardé : navier_stokes.gif")

    # ── Streamlines finales ───────────────────────────────────────────────
    @inbounds for j in 1:ny, i in 1:nx
        uc[i,j] = 0.5*(u[i,j] + u[i+1,j])
        vc[i,j] = 0.5*(v[i,j] + v[i,j+1])
    end
    plt_stream = plot_streamlines(uc, vc, xs, ys; nseeds=22, clim=U0)
    savefig(plt_stream, "streamlines_final.png")
    println("Streamlines sauvegardées : streamlines_final.png")

    return u, v, p, dx, dy, nx, ny, xs, ys
end

u, v, p, dx, dy, nx, ny, xs, ys = main()