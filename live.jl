using LinearAlgebra, SparseArrays, GLMakie
include("func.jl")

function main_live()
    nx, ny = 256, 256
    Lx, Ly = 1.0, 3.0
    dx, dy  = Lx / nx, Ly / ny
    U0      = 0.1
    rho     = 1.2
    Re      = 1000.0
    nu      = U0 * Lx / Re
    T_ref  = 10.0
    T_jet  = 70.0
    alpha  = 1e-5
    beta   = 3e-3
    g      = 9.81

    dt_diff = 0.25 * min(dx, dy)^2 / nu
    dt_conv = 0.5  * min(dx, dy) / U0
    dt_therm = 0.25 * min(dx,dy)^2 / alpha
    dt_buoy = 0.5 * min(dx,dy) / sqrt(g * beta * abs(T_jet - T_ref) * Ly)
    dt = min(dt_diff, dt_conv, dt_therm, dt_buoy)

    slot_width   = 1
    j_slot_start = nx÷2 - slot_width
    j_slot_end   = nx÷2 + slot_width
    V_jet        = U0

    println("Re=$(Re),  nu=$(round(nu, sigdigits=3)),  dt=$(round(dt, sigdigits=3))")
    println("Appuyez sur Échap dans la fenêtre pour arrêter.")

    # ── Champs ────────────────────────────────────────────────────────────────
    u      = zeros(nx+1, ny)
    v      = zeros(nx,   ny+1)
    p      = zeros(nx,   ny)
    T      = fill(T_ref, nx, ny)

    A_p    = laplacian2D(nx, ny, dx, dy)
    A_fact = factorize(SparseMatrixCSC(copy(A_p)))

    xs = collect(((1:nx) .- 0.5) .* dx)
    ys = collect(((1:ny) .- 0.5) .* dy)

    Du     = zeros(nx+1, ny);   Dv     = zeros(nx,   ny+1)
    Cu     = zeros(nx+1, ny);   Cv     = zeros(nx,   ny+1)
    gu     = zeros(nx+1, ny);   gv     = zeros(nx,   ny+1)
    u_star = zeros(nx+1, ny);   v_star = zeros(nx,   ny+1)
    bvec   = zeros(nx*ny);      pvec   = zeros(nx*ny)
    uc     = zeros(nx,   ny);   vc     = zeros(nx,   ny)
    DT     = zeros(nx,   ny);   CT     = zeros(nx,   ny)

    # ── Observables ───────────────────────────────────────────────────────────
    obs_uc    = GLMakie.Observable(zeros(Float32, nx, ny))
    obs_vc    = GLMakie.Observable(zeros(Float32, nx, ny))
    obs_speed = GLMakie.Observable(zeros(Float32, nx, ny))
    obs_T     = GLMakie.Observable(fill(Float32(T_ref), nx, ny))
    obs_step  = GLMakie.Observable(0)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = GLMakie.Figure(size=(1200, 700))

    ax1 = GLMakie.Axis(fig[1,1], title="u",       aspect=GLMakie.DataAspect(), xlabel="x", ylabel="y")
    ax2 = GLMakie.Axis(fig[1,2], title="v",       aspect=GLMakie.DataAspect(), xlabel="x")
    ax3 = GLMakie.Axis(fig[1,3], title=GLMakie.@lift("‖u‖  n=$($obs_step)"),
                                               aspect=GLMakie.DataAspect(), xlabel="x")
    ax4 = GLMakie.Axis(fig[1,4], title="T (°C)",  aspect=GLMakie.DataAspect(), xlabel="x")

    hm1 = GLMakie.heatmap!(ax1, xs, ys, obs_uc;    colormap=:RdBu,  colorrange=(-2, 2))
    hm2 = GLMakie.heatmap!(ax2, xs, ys, obs_vc;    colormap=:RdBu,  colorrange=(-2, 2))
    hm3 = GLMakie.heatmap!(ax3, xs, ys, obs_speed; colormap=:plasma, colorrange=(0.0, 4))
    hm4 = GLMakie.heatmap!(ax4, xs, ys, obs_T;     colormap=:hot,   colorrange=(0, 100))

    GLMakie.Colorbar(fig[2,1], hm1, vertical=false, tellwidth=false)
    GLMakie.Colorbar(fig[2,2], hm2, vertical=false, tellwidth=false)
    GLMakie.Colorbar(fig[2,3], hm3, vertical=false, tellwidth=false)
    GLMakie.Colorbar(fig[2,4], hm4, vertical=false, tellwidth=false)

    display(fig)

    # ── Gestion Échap ─────────────────────────────────────────────────────────
    running = Ref(true)
    GLMakie.on(GLMakie.events(fig).keyboardbutton) do event
        if event.action == GLMakie.Keyboard.press && event.key == GLMakie.Keyboard.escape
            running[] = false
        end
    end

    # ── Boucle principale ─────────────────────────────────────────────────────
    n = 0
    while running[] && isopen(fig.scene)
        n += 1

        apply_bc!(u, v, p, nx, ny, j_slot_start, j_slot_end, V_jet)

        fill!(Du, 0.0); fill!(Dv, 0.0)
        fill!(Cu, 0.0); fill!(Cv, 0.0)
        fill!(DT, 0.0); fill!(CT, 0.0)

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

        apply_bc!(u_star, v_star, p, nx, ny, j_slot_start, j_slot_end, V_jet)

        fill!(bvec, 0.0)
        @inbounds for j in 1:ny, i in 1:nx
            bvec[(j-1)*nx+i] = (rho/dt) * divergence(u_star, v_star, i, j, dx, dy)
        end
        bvec[1] = 0.0

        for i in 1:nx
            bvec[(ny-1)*nx + i] = 0.0   # bord haut p=0
        end

        pvec .= A_fact \ bvec
        p    .= reshape(pvec, nx, ny)

        gradp_u!(gu, p, dx)
        gradp_v!(gv, p, dy)
        @. u = u_star - (dt/rho)*gu
        @. v = v_star - (dt/rho)*gv

        @inbounds for j in 1:ny, i in 1:nx
            uc[i,j] = 0.5*(u[i,j] + u[i+1,j])
            vc[i,j] = 0.5*(v[i,j] + v[i,j+1])
        end

        apply_bc_T!(T, nx, ny, T_jet, j_slot_start, j_slot_end)
        flux_diff_T!(DT, alpha, T, dx, dy)
        flux_conv_T!(CT, uc, vc, T, dx, dy)
        @. T = T + dt*(DT - CT)

        if any(isnan, u) || any(isnan, v)
            println("⚠️  Divergence numérique à l'étape $n — arrêt.")
            break
        end

        if n % 20 == 0
            obs_uc[]    = Float32.(uc)
            obs_vc[]    = Float32.(vc)
            obs_speed[] = Float32.(sqrt.(uc.^2 .+ vc.^2))
            obs_T[]     = Float32.(T)
            obs_step[]  = n
            sleep(0.0001)
        end
    end

    println("\nSimulation terminée après $n étapes.")
    return u, v, p, T, xs, ys, dx, dy, nx, ny
end

u, v, p, T, xs, ys, dx, dy, nx, ny = main_live()