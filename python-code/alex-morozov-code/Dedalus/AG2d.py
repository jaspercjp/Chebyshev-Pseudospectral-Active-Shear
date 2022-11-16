import sys
import time
import numpy as np
from dedalus import public as de
from dedalus.extras import flow_tools
from numpy.random import default_rng

import logging
logger = logging.getLogger(__name__)

rng = default_rng()

########################
# Numerical Parameters

_resolution = 128
_box_size = 100.
Tmax = 90000.
dt = 1.0e-1

########################
# Bases and domain

x_basis = de.Fourier('x', _resolution, interval=(0, _box_size), dealias=3/2)
y_basis = de.Fourier('y', _resolution, interval=(0, _box_size), dealias=3/2)
domain = de.Domain([x_basis,y_basis], grid_dtype=np.float64)

########################
# Problem

problem = de.IVP(domain, variables=['Qxx', 'Qxy','omega', 'psi'])

#problem.parameters['Re'] = 0.0
problem.parameters['kappa'] = 1.0
problem.parameters['xi'] = 0.7
problem.parameters['Zeta'] = 0.1
problem.parameters['gamma'] = 0.3


problem.substitutions['ux'] = "dy(psi)"
problem.substitutions['uy'] = "-dx(psi)"

problem.substitutions['dxux'] = "dx(ux)"
problem.substitutions['dyux'] = "dy(ux)"
problem.substitutions['dxuy'] = "dx(uy)"

problem.substitutions['tmpNS'] = "dx(dx(Qxy)) - dy(dy(Qxy)) - 2*dx(dy(Qxx))"


problem.substitutions['LPL(A)'] = "dx(dx(A)) + dy(dy(A))"
problem.substitutions['Advec(A)'] = "ux*dx(A) + uy*dy(A)"

problem.substitutions['TrQ2'] = "Qxx*Qxx + Qxy*Qxy"
problem.substitutions['Hxx'] = "-(1-gamma/3.)*Qxx - 2*gamma*TrQ2*Qxx + LPL(Qxx)"
problem.substitutions['Hxy'] = "-(1-gamma/3.)*Qxy - 2*gamma*TrQ2*Qxy + LPL(Qxy)"

problem.substitutions['tmpPi1'] = "Qxx*Hxx + Qxy*Hxy"
problem.substitutions['tmpPi2'] = "Qxx*Hxy - Qxy*Hxx"


problem.substitutions['Pixx'] = "4*xi*Qxx*tmpPi1 \
    + 2*xi*gamma*TrQ2*Qxx \
    - 2*( dx(Qxx)*dx(Qxx) + dx(Qxy)*dx(Qxy) ) "

problem.substitutions['Piyy'] = "-4*xi*Qxx*tmpPi1 \
    - 2*xi*gamma*TrQ2*Qxx \
    - 2*( dy(Qxx)*dy(Qxx) + dy(Qxy)*dy(Qxy) ) "

problem.substitutions['Pixy'] = "4*xi*Qxy*tmpPi1 + 2*tmpPi2 \
    + 2*xi*gamma*TrQ2*Qxy \
    -2*( dx(Qxx)*dy(Qxx) + dx(Qxy)*dy(Qxy) ) "

problem.substitutions['Piyx'] = "4*xi*Qxy*tmpPi1 - 2*tmpPi2 \
    + 2*xi*gamma*TrQ2*Qxy \
    -2*( dx(Qxx)*dy(Qxx) + dx(Qxy)*dy(Qxy) ) "

problem.substitutions['XXX'] = "2*Qxx*dxux + Qxy*(dxuy+dyux)"

# Q equations

problem.add_equation("dt(Qxx) + (1.0-gamma/3.)*Qxx - LPL(Qxx) - xi*dxux \
    = -Advec(Qxx) - omega*Qxy - 2*xi*Qxx*XXX \
    - 2*gamma*TrQ2*Qxx")

problem.add_equation("dt(Qxy) + (1.0-gamma/3.)*Qxy - LPL(Qxy) - 0.5*xi*(dxuy+dyux) \
    = -Advec(Qxy) + omega*Qxx - 2*xi*Qxy*XXX \
    - 2*gamma*TrQ2*Qxy")

# Streamfunction - vorticity equations

#problem.substitutions['tmpNS'] = "dx(dx(Qxy)) - dy(dy(Qxy)) - 2*dx(dy(Qxx))"

# # If we have inertia
# problem.add_equation(" Re*dt(omega) - LPL(omega) \
#     + (Zeta - xi*kappa*(1.0-gamma/3.))*tmpNS + xi*kappa*LPL(tmpNS) \
#     = -Re*Advec(omega) + kappa*(  dx(dx(Piyx)) - dy(dy(Pixy)) + dx(dy(Piyy-Pixx)) )",\
#          condition="(nx != 0) or (ny != 0)")

# No inertia
problem.add_equation(" - LPL(omega) \
    + (Zeta - xi*kappa*(1.0-gamma/3.))*tmpNS + xi*kappa*LPL(tmpNS) \
    = kappa*(  dx(dx(Piyx)) - dy(dy(Pixy)) + dx(dy(Piyy-Pixx)) )",\
         condition="(nx != 0) or (ny != 0)")

problem.add_equation("psi = 0", condition="(nx == 0) and (ny == 0)")
problem.add_equation('LPL(psi) + omega = 0')


####################
# Build solver

solver = problem.build_solver(de.timesteppers.SBDF4)

solver.stop_sim_time = Tmax
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf


####################
# Initial conditions

#x, y = domain.grid(0), domain.grid(1)
Qxx = solver.state['Qxx']
#Qxx['g'] = np.sin(4*np.pi*x/_box_size)*np.sin(4*np.pi*y/_box_size)

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
noise = rng.standard_normal(gshape)[slices]
Qxx['g'] = 0.1+0.25*noise

#Load restart
#write, dt = solver.load_state('snapshots_s16.h5', -1)

####################
# Build analyser

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', iter=1000, max_writes=20)
snapshots.add_system(solver.state)

analysis1 = solver.evaluator.add_file_handler("energies",sim_dt=0.1, max_writes=1000)
analysis1.add_task("(1/2)*integ(ux*ux+uy*uy,'x','y')", name='Ek')

# Runtime monitoring properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=20)
flow.add_property("(1/2)*integ(ux*ux+uy*uy,'x','y')", name='q')

####################
# Main loop

try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        solver.step(dt)
        if (solver.iteration-1) % 20 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Kinetic energy = %f' %flow.min('q'))

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))


