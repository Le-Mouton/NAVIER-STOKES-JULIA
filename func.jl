using LinearAlgebra, SparseArrays, Plots

function laplacian2D(nx, ny, dx, dy)
    N = nx * ny
    A = spzeros(N, N)
    id(i, j) = (j - 1) * nx + i

    for j in 1:ny, i in 1:nx
        k = id(i, j)

        # if j == 1
        #     A[k, k] = 1.0
        #     continue
        # end

        if j == ny
            A[k, k] = 1.0
            continue
        end

        diag = 0.0
        if i > 1;  A[k, id(i-1,j)] = 1/dx^2; diag -= 1/dx^2; end
        if i < nx; A[k, id(i+1,j)] = 1/dx^2; diag -= 1/dx^2; end
        if j > 1;  A[k, id(i,j-1)] = 1/dy^2; diag -= 1/dy^2; end
        if j < ny; A[k, id(i,j+1)] = 1/dy^2; diag -= 1/dy^2; end
        A[k, k] = diag
    end
    return A
end

# function condition_bord!(u, v, U0=5.0)
#     u[1,:]   .= 0.0; u[end,:] .= 0.0
#     u[:,1]   .= U0;  u[:,end] .= 0.0
#     v[:,1]   .= 0.0; v[:,end] .= 0.0
#     v[1,:]   .= 0.0; v[end,:] .= 0.0
# end

function apply_bc!(u, v, p, nx, ny, j_slot_start, j_slot_end, V_jet)

    # Bord gauche 
    # u[1, :] .= u[2,:]
    # v[1, :] .= v[2,:]
    u[1, :] .= 0
    v[1, :] .= 0

    # Bord droit 
    # u[end, :] .= u[end-1,:]
    # v[end, :] .= v[end-1,:]
    u[end, :] .= 0
    v[end, :] .= 0

    # Bord bas 
    u[:, 1] .= 0.0
    v[:, 1] .= 0.0
    v[j_slot_start:j_slot_end, 1] .= V_jet

    # u[61:64 ,1] .= u[61:64 ,2]
    # v[61:64 ,1] .= 0
    
    # Bord haut 
    # u[:, end] .= 0
    # v[:, end] .= 0
    u[:,end] .= u[:,end-1]
    v[:,end] .= v[:,end-1]
end

@inline function divergence(u, v, i, j, dx, dy)
    (u[i+1, j] - u[i, j]) / dx + (v[i, j+1] - v[i, j]) / dy
end

function flux_diff!(F, nu, phi, dx, dy)
    nx, ny = size(phi)
    idx2 = 1/dx^2; idy2 = 1/dy^2

    @inbounds for j in 2:ny-1, i in 2:nx-1
        F[i,j] = nu * (
            (phi[i+1,j] - 2phi[i,j] + phi[i-1,j]) * idx2 +
            (phi[i,j+1] - 2phi[i,j] + phi[i,j-1]) * idy2
        )
    end

end

function flux_conv_u!(F, u, v, dx, dy)
    nx1, ny = size(u)
    idx = 1/dx; idy = 1/dy
    @inbounds for j in 2:ny-1, i in 2:nx1-1
        u_right = i < nx1 ? (u[i,j] + u[i+1,j]) * 0.5 : 0.0
        u_left  = i > 1   ? (u[i-1,j] + u[i,j]) * 0.5 : 0.0

        il = max(i-1,1); ir = min(i,nx1-1)

        v_top = (v[il,j+1] + v[ir,j+1]) * 0.5
        v_bot = (v[il,j]   + v[ir,j])   * 0.5

        ua_r = u_right >= 0 ? u[i,j] : (i < nx1 ? u[i+1,j] : u[i,j])
        ua_l = u_left  >= 0 ? (i > 1 ? u[i-1,j] : u[i,j]) : u[i,j]
        ua_t = v_top   >= 0 ? u[i,j] : (j < ny  ? u[i,j+1] : u[i,j])
        ua_b = v_bot   >= 0 ? (j > 1 ? u[i,j-1] : u[i,j]) : u[i,j]
        F[i,j] = (u_right*ua_r - u_left*ua_l)*idx + (v_top*ua_t - v_bot*ua_b)*idy
    end
end

function flux_conv_v!(F, u, v, dx, dy)
    nx, ny1 = size(v)
    idx = 1/dx; idy = 1/dy

    @inbounds for j in 2:ny1-1, i in 2:nx-1

        v_top = j < ny1 ? (v[i,j] + v[i,j+1]) * 0.5 : 0.0
        v_bot = j > 1   ? (v[i,j-1] + v[i,j]) * 0.5 : 0.0

        jb = max(j-1,1); jt = min(j,ny1-1)
        u_right = (u[i+1,jb] + u[i+1,jt]) * 0.5
        u_left  = (u[i,  jb] + u[i,  jt]) * 0.5

        va_t = v_top   >= 0 ? v[i,j] : (j < ny1 ? v[i,j+1] : v[i,j])
        va_b = v_bot   >= 0 ? (j > 1 ? v[i,j-1] : v[i,j]) : v[i,j]
        va_r = u_right >= 0 ? v[i,j] : (i < nx  ? v[i+1,j] : v[i,j])
        va_l = u_left  >= 0 ? (i > 1 ? v[i-1,j] : v[i,j]) : v[i,j]

        F[i,j] = (u_right*va_r - u_left*va_l)*idx + (v_top*va_t - v_bot*va_b)*idy
    end
end

#### TEMP2RATURE 

function flux_diff_T!(F, alpha, T, dx, dy)
    nx, ny = size(T)
    idx2 = 1/dx^2; idy2 = 1/dy^2
    @inbounds for j in 2:ny-1, i in 2:nx-1
        F[i,j] = alpha * (
            (T[i+1,j] - 2T[i,j] + T[i-1,j]) * idx2 +
            (T[i,j+1] - 2T[i,j] + T[i,j-1]) * idy2)
    end
end

function flux_conv_T!(F, uc, vc, T, dx, dy)
    nx, ny = size(T)
    idx = 1/dx; idy = 1/dy
    @inbounds for j in 2:ny-1, i in 2:nx-1

        u_r = (uc[i,j] + uc[min(i+1,nx),j]) * 0.5
        u_l = (uc[max(i-1,1),j] + uc[i,j]) * 0.5
        v_t = (vc[i,j] + vc[i,min(j+1,ny)]) * 0.5
        v_b = (vc[i,max(j-1,1)] + vc[i,j]) * 0.5


        Ta_r = u_r >= 0 ? T[i,j] : T[min(i+1,nx),j]
        Ta_l = u_l >= 0 ? T[max(i-1,1),j] : T[i,j]
        Ta_t = v_t >= 0 ? T[i,j] : T[i,min(j+1,ny)]
        Ta_b = v_b >= 0 ? T[i,max(j-1,1)] : T[i,j]

        F[i,j] = (u_r*Ta_r - u_l*Ta_l)*idx + (v_t*Ta_t - v_b*Ta_b)*idy
    end
end

function apply_bc_T!(T, nx, ny, T_jet, i_jet_start, i_jet_end)

    T[1,   :] .= T[2,   :]
    T[end, :] .= T[end-1, :]
    T[:, end] .= T[:, end-1]

    # T[4:14, 5:15] .= 32

    T[:, 1] .= T[:, 2]
    T[i_jet_start:i_jet_end, 1] .= T_jet

end

function gradp_u!(g, p, dx)
    nx, ny = size(p); idx = 1/dx
    @inbounds for j in 1:ny, i in 2:nx
        g[i,j] = (p[i,j] - p[i-1,j]) * idx
    end
end

function gradp_v!(g, p, dy)
    nx, ny = size(p); idy = 1/dy
    @inbounds for j in 2:ny, i in 1:nx
        g[i,j] = (p[i,j] - p[i,j-1]) * idy
    end
end

function gif_maker(uc, vc, p, T, T_ref, T_jet, xs, ys, dx, dy, n, Xg, Yg, xi, yj, step)
    speed  = sqrt.(uc.^2 .+ vc.^2)
    sp     = @views speed[xi,yj] .+ 1e-10
    Un     = @views uc[xi,yj] ./ sp .* (dx*step*0.5)
    Vn     = @views vc[xi,yj] ./ sp .* (dy*step*0.5)
    u_lim  = max(maximum(abs,uc), 1e-6)
    v_lim  = max(maximum(abs,vc), 1e-6)
    sp_lim = (0.0, max(maximum(speed), 1e-6))
    T_lim  = (0, 100)

    p1 = heatmap(xs, ys, uc'; title="u", color=:RdBu,
                 clims=(-u_lim,u_lim), aspect_ratio=1, colorbar=false, dpi=72)
    p2 = heatmap(xs, ys, vc'; title="v", color=:RdBu,
                 clims=(-v_lim,v_lim), aspect_ratio=1, colorbar=false, dpi=72)
    p3 = heatmap(xs, ys, speed'; title="‖u‖ t=$n", color=:plasma,
                 clims=sp_lim, aspect_ratio=1, colorbar=false, dpi=72)
    quiver!(p3, Xg, Yg; quiver=(vec(Un),vec(Vn)), color=:white, linewidth=0.6)
    p4 = heatmap(xs, ys, T'; title="T (°C)", color=:hot,
                 clims=T_lim, aspect_ratio=1, colorbar=true, dpi=72)

    plot(p1, p2, p3, p4; layout=(2,2), size=(900,800), dpi=72)
end

# ── Streamlines RK2 ───────────────────────────────────────────────────────
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
                u1 = interp2(uc,x,y); v1 = interp2(vc,x,y)
                sp1 = sqrt(u1^2+v1^2)+1e-12
                x2 = clamp(x+ds*u1/sp1, xmin, xmax)
                y2 = clamp(y+ds*v1/sp1, ymin, ymax)
                u2 = interp2(uc,x2,y2); v2 = interp2(vc,x2,y2)
                um = (u1+u2)*0.5; vm = (v1+v2)*0.5
                sp = sqrt(um^2+vm^2)+1e-12
                x = clamp(x+ds*um/sp, xmin, xmax)
                y = clamp(y+ds*vm/sp, ymin, ymax)
                push!(px,x); push!(py,y)
            end
            push!(lines,(px,py))
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