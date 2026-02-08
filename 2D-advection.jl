# =========================================================
# 2D Advection - Explicit upwind scheme (flux in x)
# =========================================================

using Printf
using Plots


# Add ghost cells for periodic boundary conditions
function add_ghosts(u)
    Nx, Ny = size(u)
    u_ext = zeros(Nx+2, Ny+2)
    u_ext[2:Nx+1, 2:Ny+1] = u
    return u_ext
end

# Periodic boundary condition function
function apply_periodic!(u)
    Nx, Ny = size(u)
    # in x-direction
    u[1, :] = u[Nx-1, :]
    u[Nx, :] = u[2, :]
    # in y-direction (simple copy for consistency)
    u[:, 1] = u[:, Ny-1]
    u[:, Ny] = u[:, 2]
end

# Parameters
Nx, Ny = 100, 100       # grid size
Lx, Ly = 1.0, 1.0       # domain dimensions
velocity = 1.0          # velocity in x
CFL = 0.4               # Courant number
tmax = 0.5              # final time

dx = Lx / Nx
dy = Ly / Ny
dt = CFL * dx / abs(velocity)
nt = Int(floor(tmax / dt))

# Grid and initial condition
x = range(dx/2, stop=Lx - dx/2, length=Nx)
y = range(dy/2, stop=Ly - dy/2, length=Ny)
u = zeros(Nx, Ny)

# Initial condition: Gaussian "bump" at the center
for j in 1:Ny, i in 1:Nx
    u[i,j] = exp(-50*((x[i]-0.3)^2 + (y[j]-0.5)^2))
end

u = add_ghosts(u)

# Time-stepping loop
for it in 1:nt
    apply_periodic!(u)
    u_new = copy(u)
    for j in 2:Ny+1, i in 2:Nx+1
        # Upwind in x only
        du = velocity > 0 ? (u[i,j] - u[i-1,j]) : (u[i+1,j] - u[i,j])

        # Equivalent to:
        # if velocity > 0
        #     du = u[i, j] - u[i-1, j]
        # else
        #     du = u[i+1, j] - u[i, j]
        # end

        u_new[i,j] = u[i,j] - dt*(velocity*du/dx)
    end

    u .= u_new
    # Equivalent to:
    # u[:,:] = u_new[:,:]

    if it % 50 == 0
        @printf("it=%d, t=%.3f\n", it, it*dt)
    end
end

# Remove ghost cells
u_final = u[2:end-1, 2:end-1]

# Plot contourf
p = contourf(x, y, u_final', xlabel="x", ylabel="y",
             title="2D Advection (flux in x)", color=:viridis)

savefig(p, "advection_contour.png")
@info "Figure saved → advection_contour.png"

contourf(x, y, u_final', xlabel="x", ylabel="y",
         title="2D Advection (flux in x)", color=:viridis)