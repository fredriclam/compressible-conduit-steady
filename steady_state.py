''' Computes steady state. In the steady-state, the mass equation simplifies
to a constraint on constant mass flux j = j0. The momentum equation in terms
or mixture pressure, the energy equation in terms of enthalpy, and the mass
exchange between phases are modeled in terms of three coupled ODEs.
Mass fractions of non-reacting phases are constant in the steady-state as a
consequence of the mass equation.

Sample usage:

import matplotlib.pyplot as plt
import steady_state as ss
import numpy as np
f = ss.SteadyState(1e5, 1,
  override_properties={
    "yC": 0.00
  })
x = np.linspace(-4150, -150, 1000) 
U = f(x)
for i in range (7): 
  plt.subplot(3,3,i+1)
  plt.plot(x, U[...,i])
plt.show()

'''

import numpy as np
import scipy
import scipy.integrate
import matplotlib.pyplot as plt
import compressible_conduit_steady.material_properties as matprops

# DONE: TEST: why is the limit tau_d -> Infty giving negative exsolved mass?
#  -- too much pressure loss before it reaches the top. Events p->0, h->0 added.

# WARNING: setting u may have multiple possible p_chambers. This corresponds to
# a different position of the exsolution front
# TODO: cache solution for sampling
# TODO: implement property pass-through (material parameters)
# also, use local space variables everywhere instead of param=.
# TODO: mode for solving for pchamber instead of j0
# print->log
# plot interactive checks
# matprops interactive checks, thermopotential surface plotting


class SteadyState():

  def __init__(self, p_vent:float, inlet_input_val:float, input_type:str="u",
    override_properties:dict=None):
    ''' 
    Steady state ODE solver.
    Inputs:
      p_vent: vent pressure (if small enough, expect flow choking).
      inlet_input_val: chamber inlet value of velocity u, pressure p, or mass
        flux j. Provide the corresponding type in input_type.
      input_type: "u", "p", or "j" as provided by user.
      override_properties (dict): map from property name to value. See first
        section of __init__ for overridable properties.
    Call this object to sample the solution at a grid x, consistent with the
    provided value of conduit_length.
    Providing the number of elements at initialization helps, since no
    re-evaluation of the numerical solution is required. Re-evaluation risks
    perturbing the location of a sonic boundary. A one-node correction is
    included against this case, extrapolating the value at the vent. See
    __call__ for this implementation.
    '''
    # Validate properties
    if override_properties is None:
      override_properties = {}
    self.override_properties = override_properties
    ''' Set default and overridable properties'''
    # Water mass fraction (uniformly applied in conduit)
    self.yWt            = override_properties.pop("yWt", 0.03)
    self.yA             = override_properties.pop("yA", 1e-7)
    # Water vapour presence slightly above zero for numerics in unsteady case
    #   Higher makes numerics more happy, even for steady-state exsolution
    #   1e-5 is 10 ppm
    self.yWvInletMin    = override_properties.pop("yWvInletMin", 1e-5)
    # Crystal content (mass fraction; must be less than yl = 1.0 - ywt)
    self.yC             = override_properties.pop("yC", 0.01)
    self.yCMin          = override_properties.pop("yCMin", 1e-5)
    # Critical volume fraction
    self.crit_volfrac   = override_properties.pop("crit_volfrac", 0.7)
    # Exsolution timescale
    self.tau_d          = override_properties.pop("tau_d", 1.0)
    # Viscosity (Pa s)
    self.mu             = override_properties.pop("mu", 1e5)
    # Conduit dimensions (m)
    self.conduit_radius = override_properties.pop("conduit_radius", 50)
    self.conduit_length = override_properties.pop("conduit_length", 4000)
    # Chamber conditions
    self.T_chamber      = override_properties.pop("T_chamber", 800+273.15)
    # Gas properties
    self.c_v_magma      = override_properties.pop("c_v_magma", 3e3)
    self.rho0_magma     = override_properties.pop("rho0_magma", 2.7e3)
    self.K_magma        = override_properties.pop("K_magma", 10e9)
    self.p0_magma       = override_properties.pop("p0_magma", 10e6)
    self.solubility_k   = override_properties.pop("solubility_k", 5e-6)
    self.solubility_n   = override_properties.pop("solubility_n", 0.5)
    # Number of evaluation points for ODE solver
    self.N_x            = override_properties.pop("NumElems", 200) + 1

    # Debug option (caches AinvRHS, source term F as lambdas that cannot be
    # pickled by the default pickle module)
    self._DEBUG = False

    # Input validation
    if self.yC + self.yA + self.yWt > 1:
      raise ValueError(f"Component mass fractions are [" +
        f"{self.yC, self.yA, self.yWt, 1.0-self.yC-self.yA-self.yWt}] for " +
        f"[crystal, air, water, melt].")
    if len(override_properties.items()) > 0:
      raise ValueError(
        f"Unused override properties:{list(override_properties.keys())}")

    # Set depth of conduit inlet (0 is surface)
    self.x0 = -self.conduit_length

    # Set mixture properties
    mixture = matprops.MixtureMeltCrystalWaterAir()
    mixture.magma = matprops.MagmaLinearizedDensity(c_v=self.c_v_magma,
      rho0=self.rho0_magma, K=self.K_magma, p_ref=self.p0_magma)
    mixture.k, mixture.n = self.solubility_k, self.solubility_n
    self.mixture = mixture

    # Set default mesh
    self.x_mesh = np.linspace(self.x0, 0, self.N_x)
    # Run once at the provided values
    self._set_cache(p_vent, inlet_input_val, input_type=input_type)

    # RHS function cache
    if self._DEBUG:
      self.F = None

    ''' Thermo functions in terms of (p, h, y)'''

  def T_ph(self, p, h, y):
    return self.mixture.T_ph(p, h, self.yA, y, 1.0-y-self.yA)

  def v_mix(self, p, T, y):
    return self.mixture.v_mix(p, T, self.yA, y, 1.0-y-self.yA)

  def dv_dp(self, p, T, y):
    return self.mixture.dv_dp(p, T, self.yA, y, 1.0-y-self.yA)

  def dv_dh(self, p, T, y):
    return self.mixture.dv_dh(p, T, self.yA, y, 1.0-y-self.yA)

  def dv_dy(self, p, T, y):
    return self.mixture.dv_dy(p, T, self.yA, y, 1.0-y-self.yA)

  def vf_g (self, p, T, y):
    return self.mixture.vf_g(p, T, self.yA, y, 1.0-y-self.yA)

  def x_sat(self, p):
    return self.mixture.x_sat(p)

  def y_wv_eq(self, p):
    return self.mixture.y_wv_eq(p, self.yWt, self.yC)


  def A(self, p, h, y, j0):
    ''' Returns coefficient matrix to ODE (A in A dq/dx = f(q)). '''
    A = np.zeros((3,3))
    # Evaluate mixture state
    T = self.T_ph(p, h, y)
    v = self.v_mix(p, T, y)
    # Construct coefficient matrix
    A[0,:] = [j0**2 * self.dv_dp(p,T,y) + 1.0,
              j0**2 * self.dv_dh(p,T,y),
              j0**2 * self.dv_dy(p,T,y)]
    A[1,:] = [v, -1, 0]
    A[2,:] = [0, 0, j0*v]
    return A

  ''' Define eigenvalues of A '''
  def eigA(self, p, h, y, j0):
    '''The eigenvalues are 
      u,
    which transports the chemical state (dy/dx), and the pair
      0.5 * j0^2 * (dv/dp)_{h,y} + 0.5 * sqrt((j0^2 dv/dp)^2 + 4(1 + j0^2 dv/dp + v j0^2 (dv/dh)) ),
      0.5 * j0^2 * (dv/dp)_{h,y} - 0.5 * sqrt((j0^2 dv/dp)^2 + 4(1 + j0^2 dv/dp + v j0^2 (dv/dh)) ).
    This conjugate pair replaces the usual isentropic sound speed eigenvalues.
    ''' 
    # Evaluate mixture state
    T = self.T_ph(p, h, y)
    v = self.v_mix(p, T, y)  
    # Compute eigenvalues
    u = j0 * v
    _q1 = j0**2 * self.dv_dp(p, T, y)
    _q2 = np.sqrt(_q1**2 + 4 * (1 + _q1 + v * j0**2 * self.dv_dh(p, T, y)))
    l1 = 0.5*(_q1 - _q2)
    l2 = 0.5*(_q1 + _q2)
    return np.array([u, l1, l2])
  
  def F_fric(self, p, T, y, yC, yWt, u) -> float:
    ''' Exposed friction (force per unit volume)'''
    is_frag = float(self.vf_g(p, T, y) > self.crit_volfrac)
    return -8.0*self.mu/self.conduit_radius**2 * u * (1.0 - is_frag)

  def solve_ssIVP(self, p_chamber, j0) -> tuple:
    ''' Solves initial value problem for (p,h,y)(x), given fully specified
    chamber (boundary) state.
    Returns:
      t: array
      state: array (p, h, y) (array sized 3 x ...)
      tup: informational tuple with solve_ivp return value `soln`,
        and system eigvals at vent) '''

    # Pull parameter values
    yA, yWt, yC, crit_volfrac, mu, tau_d, conduit_radius, T_chamber = \
      self.yA, self.yWt, self.yC, self.crit_volfrac, self.mu, self.tau_d, \
      self.conduit_radius, self.T_chamber
    x0, N_x = self.x0, self.N_x
    # Compute inlet conditions
    yWvInlet = np.clip(self.y_wv_eq(p_chamber), self.yWvInletMin, None)
    h_chamber = self.mixture.h_mix(
      p_chamber, T_chamber, yA, yWvInlet, 1.0-yA-yWvInlet)

    ''' Define momentum source (using captured parameters). '''
    # Define gravity momentum source
    F_grav = lambda rho: rho * (-9.8)
    # Define ODE momentum source term sum in terms of density rho
    F_rho = lambda p, T, y, rho: F_grav (rho) + self.F_fric(p, T, y, yC, yWt, j0/rho)

    ''' Define water vapour mass fraction source. '''
    # Define source in mass per total volume
    S_source = lambda p, y, rho: rho / tau_d * (1.0 - yC - yWt) * float(y > 0) * (
      (yWt - y)/(1.0 - yC - yWt) - self.x_sat(p))
    # Define equivalent source for mass fraction exsolved (y, or yWv)
    Y_unlimited = lambda p, y: 1.0 / tau_d * float(yWt > y) * (
      (yWt - y) - self.x_sat(p) * (1.0 - yC - yWt))
    # One-way gating
    Y = lambda p, y: float(y > 0) * Y_unlimited(p, y) \
      + float(y <= 0) * np.clip(Y_unlimited(p,y), 0, None)
    # Ramp gating
    # Y = lambda p, y: np.clip(y, None, self.yWvInletMin) / self.yWvInletMin * Y_unlimited(p, y)
    
    ''' Set source term vector. '''
    def F(q):
      # Unpack q of size (3,1)
      p, h, y = q
      F = np.zeros((3,1))
      # Compute mixture temperature, density
      T = self.T_ph(p, h, y)
      rho = 1.0/self.v_mix(p, T, y)
      # Compute source vector
      F[0] = F_rho(p, T, y, rho) 
      F[2] = Y(p, y)
      return F
    if self._DEBUG:  
      # Cache source term (cannot be pickled with default pickle module)
      self.F = F

    ''' Set ODE RHS A^{-1} F '''
    def AinvRHS_numinv(x, q):
      ''' Basic A^{-1} F evaluation.
      Used in ODE solver for dq/dx == RHS(x, q).
      Use AinvRHS instead for speed; this function
      shows more clearly the equation being solved,
      but relies on numerical inversion of a 3x3
      matrix.
      '''
      # Solve for RHS ode
      return np.linalg.solve(self.A(*q, j0), F(q)).flatten()

    def AinvRHS(x, q):
      ''' Precomputed A^{-1} f for speed.
      Uses block triangular inverse of
        [A b]^{-1}  = [A^{-1}  z ]
        [  u]         [       1/u]
      applied to sparse RHS vector F.
      Use this instead of RHS for speed.
      '''
      # Unpack
      p, h, y = q
      # Compute dependents
      T     = self.T_ph(p, h, y)
      # dv_dp = y * (R / p * dT_dp(p, h, y) - R * T / p**2) + (1 - y) * dvm_dp(p)
      # dv_dh = y * R / p * dT_dh(p, h, y)
      # dv_dy = (v_wv(p, T) - v_m(p)) + y * R / p * dT_dy(p, T, y)
      v     = self.v_mix(p, T, y) 
      u     = j0 * v
      # Compute first column of A^{-1}:(2,1)
      a1 = np.array([1, v]) / (1+j0**2 * self.dv_dp(p, T, y) \
        + v * j0**2 * self.dv_dh(p, T, y))
      # Compute z == -A^{-1} * b / u
      z = -j0**2 * self.dv_dy(p, T, y) / u * a1
      return Y(p, y) * np.array([*z, 1/u]) \
        + F_rho(p, T, y, 1.0/v) * np.array([*a1, 0])
    
    def RHS_reduced(x, q):
      ''' Reduced-size system for j0 == 0 case. (2x1 instead of 3x1). '''
      p, h = q
      F = np.zeros((2,1))
      # Equilibrium water vapour
      y = self.y_wv_eq(p)
      # Compute mixture temperature
      T = self.T_ph(p, h, y)
      v = self.v_mix(p, T, y)
      # Compute source vector with idempotent A^{-1} = A premultiplied
      F[0] = 1
      F[1] = v
      F *= F_rho(p, T, y, 1.0/self.v_mix(p, T, y)) 
      return F.flatten()
    
    ''' Define postprocessing eigenvalue checker '''  
    # Set captured lambdas
    T_ph, dv_dp, v_mix, dv_dh = self.T_ph, self.dv_dp, self.v_mix, self.dv_dh
    class EventChoked():
      def __init__(self):
        self.terminal = True
        self.sonic_tol = 1e-7
      def __call__(self, t, q):
        # Compute equivalent condition to conjugate pair eigenvalue == 0
        # Note that this does not check the condition u == 0 (or j0 == 0).
        p, h, y = q
        T = T_ph(p, h, y)
        # dv_dp = y * (R / p * dT_dp(p, h, y) - R * T / p**2) + (1 - y) * dvm_dp(p)
        # dv_dh = y * R / p * dT_dh(p, h, y)
        return j0**2 * (dv_dp(p, T, y) 
          + v_mix(p, T, y) * dv_dh(p, T, y)) \
          + 1.0 - self.sonic_tol
        # Default numerical eigenvalue computation
        return np.abs(np.linalg.eigvals(A(*q, j0))).min() - self.sonic_tol
    
    class ZeroPressure():
      def __init__(self):
        self.terminal = True
        self.direction = -1.0
      def __call__(self, t, q):
        p, h, y = q
        return p

    class ZeroEnthalpy():
      def __init__(self):
        self.terminal = True
        self.direction = -1.0
      def __call__(self, t, q):
        p, h, y = q
        return h
    
    class PositivePressureGradient():
      def __init__(self, RHS):
        self.terminal = True
        self.RHS = RHS
      def __call__(self, t, q):
        # Right hand side of dp/dx; is zero when dp/dx>0
        return float(self.RHS(t, q)[0] <= 0)

    # Set chamber (inlet) condition (p, h, y) with y = y_eq at pressure
    q0 = np.array([p_chamber, h_chamber, yWvInlet])

    if self._DEBUG:
      # Cache ODE details
      self.ivp_inputs = (AinvRHS, (self.x_mesh[0],self.x_mesh[-1]), q0, self.x_mesh, "Radau",
        [EventChoked(), ZeroPressure(), ZeroEnthalpy(), PositivePressureGradient(AinvRHS)])
    # Call ODE solver
    if j0 > 0:
      soln = scipy.integrate.solve_ivp(AinvRHS, (self.x_mesh[0],self.x_mesh[-1]), q0, t_eval=self.x_mesh, method="Radau",
        events=[EventChoked(), ZeroPressure(), ZeroEnthalpy(), PositivePressureGradient(AinvRHS)])
      # Output solution
      soln_state = soln.y
    else: # Exsolution length scale u * tau_d -> 0
      # Exact zero flux: use reduced (equilibrium chemistry) system
      soln = scipy.integrate.solve_ivp(RHS_reduced, (self.x_mesh[0],self.x_mesh[-1]), q0[:-1], t_eval=self.x_mesh, method="Radau",
        events=[EventChoked(), ZeroPressure(), ZeroEnthalpy(), PositivePressureGradient(RHS_reduced)])
      # Augment output solution with y at equilibrium
      soln_state = np.vstack((soln.y, self.y_wv_eq(soln.y[0,:])))

    # Compute eigenvalues at the final t
    eigvals_t_final = self.eigA(*soln_state[:,-1], j0)

    return soln.t, soln_state, (soln, eigvals_t_final)
  
  def _set_cache(self, p_vent:float, inlet_input_val:float,
    input_type:str="u"):
    _, _, calc_details = self.solve_steady_state_problem(
      p_vent, inlet_input_val, input_type=input_type, verbose=True)
    self.j0 = calc_details["j0"]
    self.p_chamber = calc_details["p"]
  
  def __call__(self, x:np.array, is_return_raw_phy:bool=False) -> np.array:
    '''Returns U sampled on x in quail format (default).
    Requires x to be consistent in length with conduit.length when
    _set_cache is called.
    ''' 
    # Save mesh as side effect
    self.x_mesh = x
    # Check that input x is consistent with internal length
    if np.abs((x.max() - x.min()) / self.conduit_length - 1.0) > 1e-7:
      raise Exception(
        f"Input x did not correspond to conduit length at initialization.")
    if is_return_raw_phy:
      # Return solution state (p, h, y)
      return self.solve_ssIVP(self.p_chamber, self.j0)[1]
    else:
      p, h, y = self.solve_ssIVP(self.p_chamber, self.j0)[1]

      # Correction for mesh perturbation (correct for up to one node)
      if len(y.ravel()) == len(self.x_mesh) - 1:
        # Extrapolate
        p, h, y = np.hstack((p, 2*p[-1] + p[-2])), \
          np.hstack((h, 2*h[-1] + h[-2])), np.hstack((y, 2*y[-1] + y[-2]))

      # Mass fraction correction
      y = np.where(y < 0, self.yWvInletMin, y)
      # Crystallinity correction
      yC = np.max((self.yC, self.yCMin))

      T = self.T_ph(p, h, y)
      v = self.v_mix(p, T, y)
      # Load and return conservative state vector
      U = np.zeros((*x.shape,8))
      U[...,0] = self.yA / v
      U[...,1] = y / v
      U[...,2] = (1.0 - y - self.yA) / v
      U[...,3] = self.j0
      U[...,4] = 0.5 * self.j0**2 * v + h/v - p
      U[...,5] = self.yWt / v
      U[...,6] = yC / v
      U[...,7] = 0.0 # Fragmented magma condition
      return U

  def solve_steady_state_problem(self, p_vent:float, inlet_input_val:float,
    input_type:str="u", verbose=False):
    ''' Solves for the choking pressure and corresponding flow state.
    Input mass flux j0, velocity u, or chamber pressure p_chamber to compute
    the corresponding steady state. Specify input_type="j", "u", or "p" to access
    these modes. Note that j, u, and p are interdependent so that only one
    is required.
    The steady state problem is solved using the shooting method for j0 or p
    against the prescribed vent pressure if the pressure is above the choking
    pressure but below the hydrostatic vent pressure.

    Devnote: The p-case can be made faster by directly solving for the choking
    pressure, and then checking if flow is choked at the computed j0.
    '''
    # Pull parameter values
    yA, yWt, yC, crit_volfrac, mu, tau_d, conduit_radius, T_chamber = \
      self.yA, self.yWt, self.yC, self.crit_volfrac, self.mu, self.tau_d, \
      self.conduit_radius, self.T_chamber
    
    p_global_min = 0.1e5
    if p_vent < p_global_min:
      raise ValueError("Vent pressure below lowest tested case (0.1 bar).")

    # Select mode
    if input_type.lower() in ("u", "u0",):
      # Set p_chamber range for finding max
      p_min, p_max = np.max((p_global_min, p_vent)), 1e9
      z_min, z_max = p_min, p_max
      # Define dependence of inlet volume on p_chamber
      v_pc = lambda p_chamber: self.mixture.v_mix(p_chamber, T_chamber, yA,
        np.clip(self.y_wv_eq(p_chamber), self.yWvInletMin, None),
        1.0 - np.clip(self.y_wv_eq(p_chamber), self.yWvInletMin, None))
      u0 = inlet_input_val
      # Define solve kernel that returns (x, (p, h, y), (soln, eigvals))
      solve_kernel = lambda p_chamber: self.solve_ssIVP(
        p_chamber, u0/v_pc(p_chamber))
      p_vent_max = p_max
      _input_type = "u"
      mass_flux_cofactor = lambda p: 1.0/v_pc(p)

      ''' Additional estimation to filter out root for fragmented magma at the inlet '''
      # Compute maximum exsolvable in conduit
      yMax = self.yWt - self.x_sat(p_vent) * (1.0 - self.yC - self.yWt - self.yA)
      # Compute maximum water vapour volume
      vwMax = 1.0 / p_vent * self.mixture.waterEx.R * self.T_chamber
      # Compute maximum mixture volume
      vMax = yMax * vwMax + (1 - yMax) * self.mixture.magma.v_pT(p_vent, None)
      # Estimate minimum chamber pressure
      p_est = p_vent + self.conduit_length * 9.8 /  vMax
      # Set arbitrary factor of minimum pressure to filter out fragmented-inlet root
      p_min = np.max((p_min, 2.0*p_est))

    elif input_type.lower() in ("j", "j0",):
      # Set p_chamber range for finding max
      p_min, p_max = np.max((p_global_min, p_vent)), 1e9
      z_min, z_max = p_min, p_max
      j0 = inlet_input_val
      # Define solve kernel that returns (x, (p, h, y), (soln, eigvals))
      solve_kernel = lambda p_chamber: self.solve_ssIVP(
        p_chamber, u0/v_pc(p_chamber))
      p_vent_max = p_max
      _input_type = "u"
      mass_flux_cofactor = lambda p: 1.0
    elif input_type.lower() in ("p", "p0",):
      # Set j0 range for finding max 
      j0_min, j0_max  = 0.0, 10e3
      z_min, z_max = j0_min, j0_max
      p_chamber = inlet_input_val
      # Define solve kernel that returns (x, (p, h, y), (soln, eigvals))
      solve_kernel = lambda j0: self.solve_ssIVP(p_chamber, j0)
      p_vent_max = calc_vent_p(j0_min)
      _input_type = "p"
    else:
      raise Exception('Unknown input_type (use "u", "j", "p").')
    
    # Define mapping z -> p_vent, where z is the conjugate to the input value
    # (user inputs u or j0: z is p; user inputs p: z is j0)
    calc_vent_p = lambda z: solve_kernel(z)[1][0][-1]
    # Define mapping z -> lambda_min(x = 0)
    def eigmin_top(z):
      ''' Returns the smaller conjugate-pair eigval at top,
      or negative value if the matrix is singular at depth.
      Assumes that the correct eigval is indexed by 1 in list
      [u, u-k, u+k].''' 
      _t, _z, outs = solve_kernel(z)
      return -1e-1 if len(outs[0].t_events[0]) != 0 or not outs[0].success else \
        np.abs(outs[1][1])

    ''' 
    For input p_chamber, the bounds on p_vent are given by the hydrostatic and
    choking j0.
    For input u or j, for low enough chamber pressure, p drops below p_vent. For
    high enough chamber pressure, the flow chokes at the vent but vent pressure
    is continuously dependent on the chamber pressure. The hydrostatic p_chamber
    provides a lower bound on p_chamber. As p_chamber increases, p_vent.
    '''
    # Solve for maximum j0 / minimum p_chamber that does not choke
    
    if _input_type == "p":
      z_choke = scipy.optimize.brenth(lambda z: eigmin_top(z), z_min, z_max)
      z_min = z_choke
      ''' Check vent flow state for given p_vent, and solve for solution [p(x), h(x), y(x)]. '''
      if p_vent < calc_vent_p(z_choke):
        # Choked case
        print("Choked at vent.")
        x, (p_soln, h_soln, y_soln), (soln, _) = solve_kernel(z_choke)
      elif p_vent > p_vent_max:
        # Inconsistent pressure (exceeds hydrostatic pressure consistent with chamber pressure)
        print("Vent pressure is too high (reverse flow required to reverse pressure gradient).")
        x, (p_soln, h_soln, y_soln), soln = None, (None, None, None), None
      else:
        print("Subsonic flow at vent. Shooting method for correct value of z.")
        z = scipy.optimize.brenth(lambda z: calc_vent_p(z) - p_vent, z_min, z_max)
        print("Solution j0 found. Computing solution.")
        # Compute solution at j0
        x, (p_soln, h_soln, y_soln), (soln, _) = solve_kernel(z)
    elif _input_type == "u":
      # z_max = z_choke
      print("Pressure matching for vent pressure at given velocity.")
      # Define wrapped objective taking into subpressurized flow
      def objective(z):
        soln = solve_kernel(z)
        # Retrieve 
        z_top = soln[0][-1]
        p_top = soln[1][0,-1]
        return p_top - p_vent if z_top >= self.x_mesh[-1] else -p_vent
      z = scipy.optimize.brenth(objective, z_min, z_max)
      x, (p_soln, h_soln, y_soln), (soln, _) = solve_kernel(z)

    if verbose:
      # Package extra details on calculation.
      calc_details = {
        "soln": soln,
      }
      if _input_type == "u":
        calc_details["p_min"] = z_min
        calc_details["p_max"] = z_max
        calc_details["p"] = z
        # Mass flux cofactor is rho if input was u rather than rho*u
        calc_details["j0"] = mass_flux_cofactor(z) * inlet_input_val
      elif _input_type == "p":
        calc_details["j0_min"] = z_min
        calc_details["j0_max"] = z_max
        calc_details["p"] = inlet_input_val
        calc_details["j0"] = z

      return x, (p_soln, h_soln, y_soln), calc_details
    else:
      return x, (p_soln, h_soln, y_soln)


if __name__ == "__main__":
  ''' Perform unit test '''

  ss = SteadyState()
  p_range = np.linspace(1e5, 10e6, 50)
  results_varp = [ss.solve_ssIVP(p_chamber=p_chamber, j0=2700*1.0) for p_chamber in p_range]
  print(results_varp[0])
  for result in results_varp:
    plt.plot(result[0], result[1][0,:], '.-')
 
  p_range = np.linspace(1e5, 10e6, 20)
  results_varp = [ss.solve_steady_state_problem(p_vent, 1.0, "u") for p_vent in p_range]
  u_range = np.linspace(0.01, 10, 10)
  results_varu = [ss.solve_steady_state_problem(1e5, u, "u") for u in u_range]

  x, (p, h, y) = ss.solve_steady_state_problem(0.5e5, 1.0, "u")
  plt.plot(x, p, '--', color="black")
  plt.show()

  ''' Plot sample solution '''
  x, (p, h, y) = ss.solve_steady_state_problem(1e5, 1.0, "u")
  plt.figure()
  plt.subplot(1,4,1)
  plt.plot(x, p, '.-')
  plt.subplot(1,4,2)
  plt.plot(x, h, '.-')
  plt.subplot(1,4,3)
  plt.plot(x, y, '.-')
  plt.subplot(1,4,4)
  phi = ss.mixture.vf_g(p, ss.mixture.T_ph(p, h, ss.yA, y, 1.0-ss.yA-y), ss.yA, y, 1.0-ss.yA-y)
  plt.plot(x, phi, '.-')
  plt.show()