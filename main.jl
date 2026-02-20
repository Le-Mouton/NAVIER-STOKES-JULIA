using LinearAlgebra, SparseArrays, Plots
include("func.jl")

function main()

    nx, ny = 64, 64
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

    u = zeros(nx+1, ny)
    v = zeros(nx,   ny+1)
    p = zeros(nx,   ny)

    A_p = laplacian2D(nx, ny, dx, dy)
    Afix = SparseMatrixCSC(copy(A_p))

    Afix[1, :] .= 0.0
    Afix[1, 1]  = 1.0
    dropzeros!(Afix)

    xs = ((1:nx) .- 0.5) .* dx
    ys = ((1:ny) .- 0.5) .* dy

    function omega_optimal(nx, ny)
        rho_J = (cos(π / nx) + cos(π / ny)) / 2
        return 2 / (1 + sqrt(1 - rho_J^2))
    end

    ω_opt = omega_optimal(nx, ny)
    println("ω optimal = $(round(ω_opt, sigdigits=5))")

    At_cached = SparseMatrixCSC(transpose(Afix))

    Du = zeros(nx+1, ny); Dv = zeros(nx, ny+1)
    Cu = zeros(nx+1, ny); Cv = zeros(nx, ny+1)
    gu = zeros(nx+1, ny); gv = zeros(nx, ny+1)
    pvec = zeros(nx*ny)

    step = max(1, nx ÷ 20)
    xi   = 1:step:nx
    yj   = 1:step:ny
    Xg   = vec([xs[i] for i in xi, j in yj])
    Yg   = vec([ys[j] for i in xi, j in yj])

    u_star = zeros(nx+1, ny)
    v_star = zeros(nx,   ny+1)
    bvec   = zeros(nx*ny) 

    anim = @animate for n in 1:nt

        print("Étape $n sur $nt \n")

        condition_bord!(u, v, U0)

        Du = flux_diff(nu, u, dx, dy)
        Dv = flux_diff(nu, v, dx, dy)
 
        Cu = flux_conv_u(u, v, dx, dy)
        Cv = flux_conv_v(u, v, dx, dy)

        u_star = u .+ dt .* (Du .- Cu)
        v_star = v .+ dt .* (Dv .- Cv)

        condition_bord!(u_star, v_star, U0)

        b = zeros(nx * ny)

        for j in 1:ny, i in 1:nx
            k = (j-1)*nx + i
            b[k] = (rho / dt) * divergence(u_star, v_star, i, j, dx, dy)
        end

        b[1] = 0.0
        bvec = copy(vec(b))
        bvec[1] = 0.0
        x0 = vec(p)

        tol_poisson = max(1e-5, 1e-4 * nu)

        pvec .= vec(p)
        gaussEidel(At_cached, bvec, pvec; tol=tol_poisson, maxiter=50_000, ω=ω_opt)
        p .= reshape(pvec, nx, ny)

        u .= u_star .- (dt / rho) .* gradp_u(p, dx)
        v .= v_star .- (dt / rho) .* gradp_v(p, dy)
        condition_bord!(u, v, U0)

        uc = 0.5 .* (u[1:end-1, :] .+ u[2:end, :])
        vc = 0.5 .* (v[:, 1:end-1] .+ v[:, 2:end])
        gif_maker(uc, vc, p, xs, ys, dx, dy, n, nt)

    end every frame_step

    gif(anim, "navier_stokes.gif", fps=15)
    println("GIF sauvegardé : navier_stokes.gif")

    # ── Streamlines finales ───────────────────────────────────────────────
    uc = 0.5 .* (u[1:end-1, :] .+ u[2:end, :])
    vc = 0.5 .* (v[:, 1:end-1] .+ v[:, 2:end])
    plt_stream = plot_streamlines(uc, vc, xs, ys; nseeds=22, clim=U0)
    savefig(plt_stream, "streamlines_final.png")
    println("Streamlines sauvegardées : streamlines_final.png")

    return u, v, p, dx, dy, nx, ny, xs, ys
end
u, v, p, dx, dy, nx, ny, xs, ys = main()