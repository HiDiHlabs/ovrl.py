### Some statistical overlap modeling convenience functions

import numpy as np
import sympy as sp


h,k,z,r = symbols('h k z r') # slice-height/2, helper-height, z, cell radius

vol_sphere = 4/3*pi*r**3
vol_cap = (pi*k**2)/3*(3*r-k) #from 0 to 2*r
area_base_cap = pi*(r**2-(r-k)**2) #from 0 to 2*r
vol_segment = simplify(integrate(area_base_cap,(k,z+r-h,z+r+h))) # segment centered at 0,from -h to h

vol_fp_sphere = 2*h*pi*r**2
vol_fp_cap = area_base_cap*h*2

# case r<h;  z=vertical position of center of ball:
# ball entirely in slice, from z=0 to z=h-r:
cell_in_slice = integrate(vol_sphere,(z,0,h-r))

# ball moving out of slice:
cell_moving_out = integrate(vol_cap,(k,0,r*2))

# the largest diameter of the sphere is inside the slice, from z=0 to z=h:
cell_max_in_slice_fp = integrate(vol_fp_sphere,(z,0,h))

# the cell is moving out of the slice:
cell_moving_out_fp = integrate(vol_fp_cap,(k,0,r*2))
cell_signal = simplify(cell_in_slice+cell_moving_out)
fp_signal = simplify(cell_max_in_slice_fp+cell_moving_out_fp)
purity_r_below_h = (cell_signal/fp_signal)*2

# case r>h;  z=vertical position of center of ball:
cell_cut_top_bottom = integrate(vol_segment,(z,0,r-h))
cell_cut_top = integrate(vol_cap,(k,0,2*h))
purity_r_above_h = simplify(cell_cut_top_bottom+cell_cut_top)/fp_signal*2

def est_ball_signal_in_slice(radius,height):
    return purity_r_below_h.subs({r:radius,h:height})