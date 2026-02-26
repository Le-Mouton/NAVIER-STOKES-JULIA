using LinearAlgebra, SparseArrays, Plots
include("func.jl")

function main()
    nx, ny = 128, 128
    Lx, Ly = 2.0, 1.0
    dx, dy  = Lx / nx, Ly / ny
    U0      = 0.5
    rho     = 1.2
    Re      = 1000.0
    nu      = U0 * Lx / Re
    dt_diff = 0.25 * min(dx, dy)^2 / nu
    dt_conv = 0.5  * min(dx, dy) / U0
    dt      = min(dt_diff, dt_conv)
    T_final = 50.0
    nt      = ceil(Int, T_final / dt)
    n_frames   = 60 
    frame_step = max(1, nt ÷ n_frames)

    slot_width = 1
    j_slot_start = nx÷2 - slot_width
    j_slot_end   = nx÷2 + slot_width
    V_jet        = U0

    T_ref   = 70.0          # température ambiante (°C)
    T_jet   = 10.0          # température du jet (°C)
    alpha   = 1e-5          # diffusivité thermique (m²/s), ~air chaud
    beta    = 3e-3           # coefficient de dilatation (1/K), ~air
    g       = 9.81           # gravité (m/s²)

    println("Re=$(Re),  nu=$(round(nu,sigdigits=3))")
    println("dt=$(round(dt,sigdigits=3)),  nt=$nt,  frame tous les $frame_step pas")

    u = zeros(nx+1, ny)
    v = zeros(nx,   ny+1)
    p = zeros(nx,   ny)

    A_p  = laplacian2D(nx, ny, dx, dy)
    Afix = SparseMatrixCSC(copy(A_p))
    A_fact = factorize(Afix)

    xs = ((1:nx) .- 0.5) .* dx
    ys = ((1:ny) .- 0.5) .* dy

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
    T      = fill(T_ref, nx, ny)
    DT     = zeros(nx, ny)
    CT     = zeros(nx, ny)
    T_star = zeros(nx, ny)

    tol_poisson = max(1e-4, 1e-3 * nu)

    step = max(1, nx ÷ 20)
    xi   = 1:step:nx
    yj   = 1:step:ny
    Xg   = vec([xs[i] for i in xi, j in yj])
    Yg   = vec([ys[j] for i in xi, j in yj])

    anim = @animate for n in 1:nt
        n % 10 == 0 && print("\rÉtape $n / $nt")

        apply_bc!(u, v, p, nx, ny, j_slot_start, j_slot_end, V_jet)

        flux_diff!(Du, nu, u, dx, dy)
        flux_diff!(Dv, nu, v, dx, dy)
        flux_conv_u!(Cu, u, v, dx, dy)
        flux_conv_v!(Cv, u, v, dx, dy)

        @. u_star = u + dt*(Du - Cu)
        @. v_star = v + dt*(Dv - Cv)

        @inbounds for j in 2:ny-1, i in 2:nx-1
            T_face = 0.5*(T[i,j] + T[i, max(j-1,1)])
            v_star[i,j] += dt * g * beta * (T_face - T_ref)
        end

        fill!(bvec, 0.0)
        @inbounds for j in 1:ny, i in 1:nx
            bvec[(j-1)*nx+i] = (rho/dt) * divergence(u_star, v_star, i, j, dx, dy)
        end
        bvec[1] = 0.0 
    
        # # Bord gauche   (i=1,  j=1..ny) → k = (j-1)*nx + 1
        # for j in 1:ny
        #     bvec[(j-1)*nx + 1] = 0.0
        # end

        # # Bord droit    (i=nx, j=1..ny) → k = (j-1)*nx + nx
        # for j in 1:ny
        #     bvec[(j-1)*nx + nx] = 0.0
        # end

        # # Bord haut
        # for i in 1:nx
        #     bvec[(ny-1)*nx + i] = 0.0           # bord haut p=0
        # end

        # Bord bas
        # for i in (nx-5):nx
        #     bvec[(i-1)*nx + 1] = 0.0           # bord haut p=0
        # end

        for i in 61:64
            bvec[i] = 0.0
        end

        pvec .= A_fact \ bvec

        p .= reshape(pvec, nx, ny)

        gradp_u!(gu, p, dx)
        gradp_v!(gv, p, dy)
        @. u = u_star - (dt/rho)*gu
        @. v = v_star - (dt/rho)*gv
        apply_bc!(u, v, p, nx, ny, j_slot_start, j_slot_end, V_jet)
        #condition_bord!(u, v, U0)

        @inbounds for j in 1:ny, i in 1:nx
            uc[i,j] = 0.5*(u[i,j] + u[i+1,j])
            vc[i,j] = 0.5*(v[i,j] + v[i,j+1])
        end

        apply_bc_T!(T, nx, ny, T_jet, j_slot_start, j_slot_end)
        flux_diff_T!(DT, alpha, T, dx, dy)
        flux_conv_T!(CT, uc, vc, T, dx, dy)
        @. T = T + dt*(DT - CT)

        gif_maker(uc, vc, p, T, T_ref, T_jet, xs, ys, dx, dy, n, Xg, Yg, xi, yj, step)

    end every frame_step

    println()
    gif(anim, "navier_stokes.gif", fps=15)
    println("GIF sauvegardé : navier_stokes.gif")

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