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
for i in range (8): 
  plt.subplot(3,3,i+1)
  plt.plot(x, U[...,i])
plt.show()

'''

import numpy as np
import scipy
import scipy.integrate
from scipy.special import erf
import matplotlib.pyplot as plt

try:
  import material_properties as matprops
except ModuleNotFoundError:
  import compressible_conduit_steady.material_properties as matprops

# DONE: TEST: why is the limit tau_d -> Infty giving negative exsolved mass?
#  -- too much pressure loss before it reaches the top. Events p->0, h->0 added.

# TODO: print->log
# TODO: plot interactive checks
# TODO in material_properties: interactive checks, thermopotential surface plotting


class SteadyState():

  # Global caching: if checkhash and pressure match, re-use value
  checkhash = 0
  last_crit_inputs = np.array([0, 0, 0, 0])
  cached_j0_p = (None, None)

  def __init__(self, x_global:np.array, p_vent:float, inlet_input_val:float,
    input_type:str="u", override_properties:dict=None,
    use_static_cache:bool=False):
    ''' 
    Steady state ODE solver.
    Inputs:
      x_global: global x coordinates (necessary to coordinate solution between
        several 1D patches)
      p_vent: vent pressure (if small enough, expect flow choking).
      inlet_input_val: chamber inlet value of velocity u, pressure p, or mass
        flux j. Provide the corresponding type in input_type.
      input_type: "u", "p", or "j" as provided by user.
      override_properties (dict): map from property name to value. See first
        section of __init__ for overridable properties.
      use_static_cache: If use_static_cache is True, inlet condition
        determination is not done again if the solution was already computed
        by any instance of SteadyState and the existing checkhash and
        critical inputs match the saved run.
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
    self.override_properties = override_properties.copy()
    ''' Set default and overridable properties'''
    # Water mass fraction (uniformly applied in conduit)
    self.yWt            = self.override_properties.pop("yWt", 0.03)
    self.yA             = self.override_properties.pop("yA", 1e-7)
    # Water vapour presence slightly above zero for numerics in unsteady case
    #   Higher makes numerics more happy, even for steady-state exsolution
    #   1e-5 is 10 ppm
    self.yWvInletMin    = self.override_properties.pop("yWvInletMin", 1e-5)
    # Crystal content (mass fraction; must be less than yl = 1.0 - ywt)
    self.yC             = self.override_properties.pop("yC", 1e-7)
    self.yCMin          = self.override_properties.pop("yCMin", 1e-7)
    # Inlet fragmented mass fraction
    self.yFInlet        = self.override_properties.pop("yFInlet", 0.0)
    # Critical volume fraction
    self.crit_volfrac   = self.override_properties.pop("crit_volfrac", 0.7)
    # Exsolution timescale
    self.tau_d          = self.override_properties.pop("tau_d", 1.0)
    # Fragmentation timescale
    self.tau_f          = self.override_properties.pop("tau_f", 1.0)
    # Viscosity (Pa s)
    self.mu             = self.override_properties.pop("mu", 1e5)
    # Conduit dimensions (m)
    self.conduit_radius = self.override_properties.pop("conduit_radius", 50)
    # Chamber conditions
    self.T_chamber      = self.override_properties.pop("T_chamber", 800+273.15)
    # Gas properties
    self.c_v_magma      = self.override_properties.pop("c_v_magma", 3e3)
    self.rho0_magma     = self.override_properties.pop("rho0_magma", 2.7e3)
    self.K_magma        = self.override_properties.pop("K_magma", 10e9)
    self.p0_magma       = self.override_properties.pop("p0_magma", 5e6)
    self.solubility_k   = self.override_properties.pop("solubility_k", 5e-6)
    self.solubility_n   = self.override_properties.pop("solubility_n", 0.5)
    # Whether to neglect deformation energy in magma EOS
    self.neglect_edfm   = self.override_properties.pop("neglect_edfm", False)

    # Debug option (caches AinvRHS, source term F as lambdas that cannot be
    # pickled by the default pickle module)
    self._DEBUG = False

    self.use_static_cache = use_static_cache

    # Output mesh
    self.x_mesh = x_global.copy()
    # Internal computation mesh
    self.x_mesh_native = x_global.copy()

    self.conduit_length = x_global.max() - x_global.min()
    # Compute liquid melt fraction
    self.yL = 1.0 - (self.yC + self.yA + self.yWt)

    # Input validation
    if self.yC + self.yA + self.yWt > 1:
      raise ValueError(f"Component mass fractions are [" +
        f"{self.yC, self.yA, self.yWt, self.yL}] for " +
        f"[crystal, air, water, melt].")
    if self.yFInlet > self.yL:
      raise ValueError(f"Inlet fragmented mass fraction exceeds the liquid" +
      f"melt mass fraction: inlet fragmented {self.yFInlet}, liquid melt {self.yL}.")
    if len(self.override_properties.items()) > 0:
      raise ValueError(
        f"Unused override properties:{list(self.override_properties.keys())}")

    # Set depth of conduit inlet
    self.x0 = x_global.min()

    # Set mixture properties
    mixture = matprops.MixtureMeltCrystalWaterAir()
    mixture.magma = matprops.MagmaLinearizedDensity(c_v=self.c_v_magma,
      rho0=self.rho0_magma, K=self.K_magma,
      p_ref=self.p0_magma, neglect_edfm=self.neglect_edfm)
    mixture.k, mixture.n = self.solubility_k, self.solubility_n
    self.mixture = mixture

    # Set tolerance for numerical solve
    self.brent_atol = 1e-5

    # Check static cache using hash of unpopped dict override_properties
    propshash = hash(tuple(override_properties.items()))
    inputs_array = np.array([x_global.min(), x_global.max(), 
      p_vent, inlet_input_val])
    if use_static_cache \
      and propshash == SteadyState.checkhash \
      and np.all(inputs_array == SteadyState.last_crit_inputs):
      self.j0, self.p_chamber = SteadyState.cached_j0_p
    else:
      SteadyState.checkhash = propshash
      SteadyState.last_crit_inputs = inputs_array.copy()
      # Run once at the provided values
      self._set_cache(p_vent, inlet_input_val, input_type=input_type)
      SteadyState.cached_j0_p = (self.j0, self.p_chamber)

    # RHS function cache
    if self._DEBUG:
      self.F = None

  ''' Define partially evaluated thermo functions in terms of (p, h, y)'''

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

  def A(self, p, h, y, yf, j0):
    ''' Return coefficient matrix to ODE (A in A dq/dx = f(q)).
    Coefficient matrix has block structure:
    [_ _ _ 0] [p ]
    [_ _ 0 0] [h ]
    [0 0 _ 0] [y ]
    [0 0 0 _] [yf]
     '''
    A = np.zeros((4,4))
    # Evaluate mixture state
    T = self.T_ph(p, h, y)
    v = self.v_mix(p, T, y)
    # Construct coefficient matrix
    A[0,:] = [j0**2 * self.dv_dp(p,T,y) + 1.0,
              j0**2 * self.dv_dh(p,T,y),
              j0**2 * self.dv_dy(p,T,y),
              0]
    A[1,:] = [v, -1, 0, 0]
    A[2,:] = [0, 0, j0*v, 0]
    A[3,:] = [0, 0, 0, j0*v]
    return A

  def eigA(self, p, h, y, yF, j0):
    ''' Return array of eigenvalues of A, which consists of the pair
      l1 = 0.5 * j0^2 * (dv/dp)_{h,y} -
        0.5 * sqrt((j0^2 dv/dp)^2 + 4(1 + j0^2 dv/dp + v j0^2 (dv/dh)) ),
      l2 = 0.5 * j0^2 * (dv/dp)_{h,y} +
        0.5 * sqrt((j0^2 dv/dp)^2 + 4(1 + j0^2 dv/dp + v j0^2 (dv/dh)) ),
    and
      u, u,
    which transport the chemical state (dy/dx). The conjugate pair eigenvalues
    replace the usual isentropic sound speed eigenvalues.
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
    return np.array([l1, l2, u, u])
  
  def F_fric(self, p, T, y, yF, rho, u) -> float:
    ''' Friction (force per unit volume)'''

    # Poll friction model
    # mu = self.mu
    mu = self.F_fric_viscosity_model(T, y, yF)

    # Compute fractional indicator using yF / yM (liquid phase, not liquid melt)
    yM = 1.0 - self.yA - y
    # Continuous alternative to float(self.vf_g(p, T, y) > self.crit_volfrac)

    frag_factor = np.clip(1.0 - yF/yM, 0.0, 1.0)
    return -8.0*mu/self.conduit_radius**2 * u * frag_factor

  def F_fric_viscosity_model(self, T, y, yF):
    ''' Calculates the viscosity as a function of dissolved
    water and crystal content (assumes crystal phase is incompressible)/
    Does not take into account fragmented vs. not fragmented (avoiding
    double-dipping the effect of fragmentation).
    '''
    # Calculate pure melt viscosity (Hess & Dingwell 1996)
    yWd = self.yWt - y
    yL = self.yL
    yM = 1.0 - y - self.yA
    mfWd = yWd / yL # mass concentration of dissolved water
    mfWd = np.where(mfWd <= 0.0, 1e-8, mfWd)
    log_mfWd = np.log(mfWd*100)
    log10_vis = -3.545 + 0.833 * log_mfWd
    log10_vis += (9601 - 2368 * log_mfWd) / (T - 195.7 - 32.25 * log_mfWd)
    # Prevent overflowing float
    log10_vis = np.where(log10_vis > 300, 300, log10_vis)
    meltVisc = 10**log10_vis
    # Calculate relative viscosity due to crystals (Costa 2005).
    alpha = 0.999916
    phi_cr = 0.673
    gamma = 3.98937
    delta = 16.9386
    B = 2.5
    # Compute volume fraction of crystal at equal phasic densities
    # Using crystal volume per (melt + crystal + dissolved water) volume
    phi_ratio = np.clip((self.yC / yM) / phi_cr, 0.0, None)
    erf_term = erf(
      np.sqrt(np.pi) / (2 * alpha) * phi_ratio * (1 + phi_ratio**gamma))
    crysVisc = (1 + phi_ratio**delta) * ((1 - alpha * erf_term)**(-B * phi_cr))
    
    viscosity = meltVisc * crysVisc
    return viscosity

  def solve_ssIVP(self, p_chamber, j0, dense_output=False) -> tuple:
    ''' Solves initial value problem for (p,h,y)(x), given fully specified
    chamber (boundary) state.
    Returns:
      t: array
      state: array (p, h, y) (array sized 3 x ...)
      tup: informational tuple with solve_ivp return value `soln`,
        and system eigvals at vent) '''

    # Pull parameter values
    yA, yWt, yC, crit_volfrac, mu, tau_d, tau_f, conduit_radius, T_chamber = \
      self.yA, self.yWt, self.yC, self.crit_volfrac, self.mu, self.tau_d, \
      self.tau_f, self.conduit_radius, self.T_chamber
    yFInlet = self.yFInlet
    # Compute auxiliary inlet conditions
    yWvInlet = np.clip(self.y_wv_eq(p_chamber), self.yWvInletMin, None)
    h_chamber = self.mixture.h_mix(
      p_chamber, T_chamber, yA, yWvInlet, 1.0-yA-yWvInlet)

    ''' Define momentum source (using captured parameters). '''
    # Define gravity momentum source
    F_grav = lambda rho: rho * (-9.8)
    # Define ODE momentum source term sum in terms of density rho
    F_rho = lambda p, T, y, yF, rho: F_grav(rho) \
      + self.F_fric(p, T, y, yF, rho, j0/rho)

    ''' Define water vapour mass fraction source. '''
    # Define source in mass per total volume
    S_source = lambda p, y, rho: rho / tau_d * (1.0 - yC - yWt - yA) * float(y > 0) * (
      (yWt - y)/(1.0 - yC - yWt) - self.x_sat(p))
    # Define equivalent source for mass fraction exsolved (y, or yWv)
    target_yWd = lambda p: np.clip(
      self.x_sat(p) * (1.0 - yC - yWt - yA), 0, yWt - self.yWvInletMin)
    Y = lambda p, y: 1.0 / tau_d * ((yWt - y) - target_yWd(p))
    # One-way gating
    # Y = lambda p, y: float(y > 0) * Y_unlimited(p, y) \
    #   + float(y <= 0) * np.clip(Y_unlimited(p,y), 0, None)
    # Ramp gating
    # Y = lambda p, y: np.clip(y, None, self.yWvInletMin) / self.yWvInletMin * Y_unlimited(p, y)
    
    ''' Set source term vector. '''
    def F(q):
      # Unpack q of size (4,1)
      p, h, y, yF = q
      F = np.zeros((4,1))
      # Compute mixture temperature, density
      T = self.T_ph(p, h, y)
      rho = 1.0/self.v_mix(p, T, y)
      # Compute (constant) liquid melt fraction
      yL = 1.0 - self.yWt - self.yC - self.yA
      yM = 1.0 - self.yA - y
      # Compute source vector
      F[0] = F_rho(p, T, y, yF, rho) 
      F[2] = Y(p, y)
      F[3] = (yM - yF) / self.tau_f * float(self.vf_g(p, T, y) >= self.crit_volfrac)
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
      but relies on numerical inversion of a 4x4
      matrix.
      '''
      # Solve for RHS ode
      return np.linalg.solve(self.A(*q, j0), F(q)).flatten()

    def AinvRHS(x, q, vectorized=False):
      ''' Precomputed A^{-1} f for speed.
      Uses block triangular inverse of
        [A b]^{-1}  = [A^{-1}  z ]
        [  u]         [       1/u]
      applied to sparse RHS vector F.
      Use this instead of RHS for speed. Supports vectorized input if
        vectorized=True
      '''
      # Unpack
      p, h, y, yF = q
      # Compute dependents
      T     = self.T_ph(p, h, y)
      # dv_dp = y * (R / p * dT_dp(p, h, y) - R * T / p**2) + (1 - y) * dvm_dp(p)
      # dv_dh = y * R / p * dT_dh(p, h, y)
      # dv_dy = (v_wv(p, T) - v_m(p)) + y * R / p * dT_dy(p, T, y)
      v     = self.v_mix(p, T, y) 
      u     = j0 * v
      # Compute first column of A^{-1}:(2,1)
      a1 = np.vstack((np.ones_like(v), v)) / (1.0+j0*j0 * self.dv_dp(p, T, y) \
        + v * j0*j0 * self.dv_dh(p, T, y))
      # Compute z == -A^{-1} * b / u
      z = -j0*j0 * self.dv_dy(p, T, y) / u * a1
      # yL = 1.0 - yWt - yC - yA
      yM = 1.0 - y - yA

      vec_length = p.shape[-1] if len(p.shape) > 0 else 1
      # return Y(p, y) * np.array([*z, 1/u, 0]) \
      #   + F_rho(p, T, y, yF, 1.0/v) * np.array([*a1, 0, 0]) \
      #   + np.array([0, 0, 0, (yM - yF) / (u * self.tau_f)
      #     * float(self.vf_g(p, T, y) >= self.crit_volfrac)])
      out = Y(p,y) * np.vstack((z,  1.0/u, np.zeros_like(u))) \
        + F_rho(p, T, y, yF, 1.0/v) \
          * np.vstack((a1, np.zeros((2, vec_length)))) \
        + np.vstack([np.zeros((3, vec_length)),
          (yM - yF) / (u * self.tau_f)
            * np.array(self.vf_g(p, T, y) >= self.crit_volfrac).astype(float)])
      if not vectorized:
        # Return flattened version
        return out.squeeze(axis=-1)
      return out
    
    def RHS_reduced(x, q):
      ''' Reduced-size system for j0 == 0 case. (2x1 instead of 4x1).
      Length scale of exsolution and fragmentation -> 0. '''
      p, h = q
      F = np.zeros((2,1))
      # Equilibrium water vapour
      y = self.y_wv_eq(p)
      # Compute mixture temperature
      T = self.T_ph(p, h, y)
      v = self.v_mix(p, T, y)
      # Compute fragmented mass fraction
      yM = 1.0 - y - yA
      yF = yM if self.vf_g(p, T, y) >= self.crit_volfrac else 0
      # Compute source vector with idempotent A^{-1} = A premultiplied
      F[0] = 1
      F[1] = v
      F *= F_rho(p, T, y, yF, 1.0/self.v_mix(p, T, y)) 
      return F.flatten()
    
    ''' Define postprocessing eigenvalue checker '''  
    # Set captured lambdas
    T_ph, dv_dp, v_mix, dv_dh = self.T_ph, self.dv_dp, self.v_mix, self.dv_dh
    class EventChoked():
      def __init__(self, y_wv_eq=None):
        self.terminal = True
        self.sonic_tol = 1e-7
        # Capture function p -> y_wv if provided
        self.y_wv_eq = y_wv_eq
      def __call__(self, t, q):
        # Compute equivalent condition to conjugate pair eigenvalue == 0
        # Note that this does not check the condition u == 0 (or j0 == 0).
        if len(q) > 2:
          p, h, y, yF = q
        else:
          p, h = q
          y = self.y_wv_eq(p)
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
        return q[0] # p

    class ZeroEnthalpy():
      def __init__(self):
        self.terminal = True
        self.direction = -1.0
      def __call__(self, t, q):
        return q[1] # h
    
    class PositivePressureGradient():
      def __init__(self, RHS):
        self.terminal = True
        self.RHS = RHS
      def __call__(self, t, q):
        # Right hand side of dp/dx; is zero when dp/dx>0
        return float(self.RHS(t, q)[0] <= 0)

    # Set chamber (inlet) condition (p, h, y) with y = y_eq at pressure
    q0 = np.array([p_chamber, h_chamber, yWvInlet, yFInlet])

    if self._DEBUG:
      # Cache ODE details
      self.ivp_inputs = (AinvRHS, (self.x_mesh[0],self.x_mesh[-1]), q0, self.x_mesh, "Radau",
        [EventChoked(), ZeroPressure(), ZeroEnthalpy(), PositivePressureGradient(AinvRHS)])
    # Call ODE solver
    if j0 > 0:
      soln = scipy.integrate.solve_ivp(AinvRHS,
        (self.x_mesh[0],self.x_mesh[-1]),
        q0,
        # t_eval=self.x_mesh,
        method="Radau", dense_output=dense_output,
        events=[EventChoked(), ZeroPressure(),
          ZeroEnthalpy(), PositivePressureGradient(AinvRHS)])
      # Output solution
      soln_state = soln.y
    else: # Exsolution length scale u * tau_d -> 0
      # Exact zero flux: use reduced (equilibrium chemistry) system
      soln = scipy.integrate.solve_ivp(RHS_reduced,
        (self.x_mesh[0],self.x_mesh[-1]),
        q0[0:2],
        t_eval=self.x_mesh,
        method="Radau", dense_output=dense_output,
        events=[EventChoked(y_wv_eq=self.y_wv_eq), ZeroPressure(),
          ZeroEnthalpy(), PositivePressureGradient(RHS_reduced)])
      # Augment output solution with y at equilibrium and yF based on fragmentation criterion
      p = soln.y[0,:]
      yWv = self.y_wv_eq(p)
      yM = 1.0 - yWv - self.yA
      T = self.T_ph(p, soln.y[1,:], yWv)
      yF = yM.copy()
      yF = np.where(self.vf_g(p, T, yWv) >= self.crit_volfrac, yM, 0.0)
      soln_state = np.vstack((soln.y, yWv, yF))

    # Compute eigenvalues at the final t
    eigvals_t_final = self.eigA(*soln_state[:,-1], j0)

    return soln.t, soln_state, (soln, eigvals_t_final)
  
  def _set_cache(self, p_vent:float, inlet_input_val:float,
    input_type:str="u"):
    _, _, calc_details = self.solve_steady_state_problem(
      p_vent, inlet_input_val, input_type=input_type, verbose=True)
    self.j0 = calc_details["j0"]
    self.p_chamber = calc_details["p"]
  
  def __call__(self, x:np.array, io_format:str="quail") -> np.array:
    '''Returns U sampled on x in quail format (default).
    Requires x to be points in interval [self.x_mesh.min(), self.x_mesh.max()].
    Inputs:
      x: array of points. If io_format=="quail", x is expected to have
        three-dimensional shape (ne, n).
      io_format: either "quail" or "phy". The latter is the native ODE solver
        output in (p, h, y, yF) space. Here y 
    ''' 
    # Check that input x is consistent with internal length
    if x.max() > self.x_mesh.max() \
      or x.min() < self.x_mesh.min():
      raise ValueError("Requested values at x not in initial global mesh.")

    # Solve steady state IVP with native mesh and precomputed mass flux
    soln = self.solve_ssIVP(self.p_chamber, self.j0, dense_output=True)
    # Extract interpolator from scipy.integrate.solve_ivp
    dense_soln = soln[2][0].sol
    # Evaluate solution using interpolator
    Q = dense_soln(np.unique(x))
    # Extrapolate out-of-bounds values using nearest value
    last_legit_index = len(np.unique(x)) \
      - np.argmax(np.unique(x)[::-1] <= soln[0].max()) - 1
    Q[:, np.unique(x) > soln[0].max()] = \
      Q[:, last_legit_index:last_legit_index+1]

    # Compute solution in requested format
    if "phy".casefold() == io_format.casefold() \
       or "native".casefold() == io_format.casefold():
      # Return solution state (p, h, y)
       # Evaluate solution using interpolator
      return Q
    elif "quail".casefold() == io_format.casefold():
      p, h, y, yF = Q
      # Mass fraction correction
      y = np.where(y < 0, self.yWvInletMin, y)
      # Crystallinity correction
      yC = np.max((self.yC, self.yCMin))
      # Compute mixture intermediates
      T = self.T_ph(p, h, y)
      v = self.v_mix(p, T, y)
      # Load and return conservative state vector
      U = np.zeros((*np.unique(x).shape,8))
      U[...,0] = self.yA / v
      U[...,1] = y / v
      U[...,2] = (1.0 - y - self.yA) / v
      U[...,3] = self.j0
      U[...,4] = 0.5 * self.j0**2 * v + h/v - p
      U[...,5] = self.yWt / v
      U[...,6] = yC / v
      U[...,7] = yF / v

      ''' Extract only values of U that correspond to query locations x. '''
      # Define associative map from value of x to state vector U
      vals = {x: U[i,:] for i, x in enumerate(np.unique(x))}
      U_out = np.zeros((*x.shape[:2],8))
      # Map sample locations to state values U
      for i in range(U_out.shape[0]):
        for j in range(U_out.shape[1]):
          U_out[i,j,:] = vals[x[i,j,0]]
      return U_out
    else:
      raise ValueError(f"Unknown output format string '{io_format}'.")

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
    yA, yWt, yC, crit_volfrac, mu, tau_d, tau_f, conduit_radius, T_chamber = \
      self.yA, self.yWt, self.yC, self.crit_volfrac, self.mu, self.tau_d, \
      self.tau_f, self.conduit_radius, self.T_chamber
    
    p_global_min = 0.1e5
    if p_vent < p_global_min:
      raise ValueError("Vent pressure below lowest tested case (0.1 bar).")

    # Select mode
    if input_type.lower() in ("u", "u0",):
      # Set p_chamber range for finding max
      p_min, p_max = np.max((p_global_min, p_vent)), 1e9
      z_min, z_max = p_min, p_max
      # Define dependence of inlet volume on p_chamber
      v_pc = lambda p_chamber: self.v_mix(p_chamber, self.T_chamber,
          np.clip(self.y_wv_eq(p_chamber), self.yWvInletMin, None))
      
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
      # Compute saturation pressure
      p_sat = (self.yWt / self.yL / self.solubility_k) ** (1/self.solubility_n)
      p_min = np.max((p_min, p_sat))
      # print(p_est, p_min, p_sat)
      z_min = p_min

    elif input_type.lower() in ("j", "j0",):
      # Set p_chamber range for finding max
      p_min, p_max = np.max((p_global_min, p_vent)), 1e9
      z_min, z_max = p_min, p_max
      j0 = inlet_input_val
      # Define solve kernel that returns (x, (p, h, y), (soln, eigvals))
      solve_kernel = lambda p_chamber: self.solve_ssIVP(
        p_chamber, j0)
      p_vent_max = p_max
      _input_type = "u"
      mass_flux_cofactor = lambda p: 1.0
    elif input_type.lower() in ("p", "p0",):
      # Set j0 range for finding max 
      j0_min, j0_max  = 0.0, 2.7e3*100
      z_min, z_max = j0_min, j0_max
      p_chamber = inlet_input_val
      # Define solve kernel that returns (x, (p, h, y), (soln, eigvals))
      solve_kernel = lambda j0: self.solve_ssIVP(p_chamber, j0)
      p_vent_max = solve_kernel(j0_min)[1][0][-1]
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
      [u-k, u+k, u, u].''' 
      _t, _z, outs = solve_kernel(z)
      return -1e-1 if len(outs[0].t_events[0]) != 0 or not outs[0].success else \
        np.abs(outs[1][0])

    ''' 
    For input p_chamber, the bounds on p_vent are given by the hydrostatic and
    choking j0.
    For input u or j, for low enough chamber pressure, p drops below p_vent. For
    high enough chamber pressure, the flow chokes at the vent but vent pressure
    is continuously dependent on the chamber pressure. The hydrostatic p_chamber
    provides a lower bound on p_chamber. As p_chamber increases, p_vent.
    '''
    # Solve for maximum j0 / minimum p_chamber that does not choke
    
    brent_atol = self.brent_atol
    if _input_type == "p":
      z_choke = scipy.optimize.brentq(lambda z: eigmin_top(z),
        z_min, z_max, xtol=brent_atol)
      z_min = z_choke
      ''' Check vent flow state for given p_vent, and solve for solution
      [p(x), h(x), y_i(x)] where y_i are the mass fractions. '''
      if p_vent < calc_vent_p(z_choke):
        # Choked case
        print("Choked at vent.")
        # Solve with one-sided precision to ensure that the last node is
        # evaluable (i.e., choking position is >= top). This is not a guarantee
        # when solution is requested 
        z = z_choke - 2*brent_atol
        x, (p_soln, h_soln, y_soln, yF_soln), (soln, _) = \
          solve_kernel(z)
      elif p_vent > p_vent_max:
        # Inconsistent pressure (exceeds hydrostatic pressure consistent with chamber pressure)
        print("Vent pressure is too high (reverse flow required to reverse pressure gradient).")
        x, (p_soln, h_soln, y_soln, yF_soln), soln = None, (None, None, None), None
      else:
        print("Subsonic flow at vent. Shooting method for correct value of z.")
        z = scipy.optimize.brentq(lambda z: calc_vent_p(z) - p_vent, z_min, z_max, xtol=brent_atol)
        print("Solution j0 found. Computing solution.")
        # Compute solution at j0
        x, (p_soln, h_soln, y_soln, yF_soln), (soln, _) = solve_kernel(z)
    elif _input_type == "u":
      # Number of times to double pressure while searching for choking pressure
      N_doubling = 14

      # Express mass flux j0 given u
      if input_type.lower() in ("j", "j0",):        
        j0_u = lambda p: j0
      else:
        u = inlet_input_val
        j0_u = lambda p: u / self.v_mix(p, self.T_chamber,
          np.clip(self.y_wv_eq(p), self.yWvInletMin, None))

      print("Computing lower bound on pressure given domain length.")
      ''' Compute loose lower bound on pressure due to gravity. '''
      # Lower bound pressure
      yMax = self.yWt - self.x_sat(p_vent) * (
        1.0 - self.yC - self.yWt - self.yA)
      # Compute maximum water vapour volume
      vwMax = 1.0 / p_vent * self.mixture.waterEx.R * self.T_chamber
      # Compute maximum mixture volume
      vMax = yMax * vwMax + (1 - yMax) * self.mixture.magma.v_pT(p_vent, None)
      p_minbound = p_vent + self.conduit_length * 9.8 /  vMax

      ''' Find lowest-pressure continuous solution '''
      _use_legacy_bound_method = False
      if _use_legacy_bound_method:
        # Arbitrarily set minimum pressure
        # approx_pseudogas_scale_height = (self.yWt * self.mixture.waterEx.R
        #   * self.T_chamber) / 9.8
        # p_guess = p_vent * np.exp(self.conduit_length /
        #   approx_pseudogas_scale_height)
        # p0 = 10*p_guess
        p0 = p_minbound
        k_last = None
        for k in range(N_doubling):
          # Compute IVP solution
          
          p_chamber = p0*2**k
          j0 = j0_u(p_chamber)
          _out = self.solve_ssIVP(p0*2**k, j0)
          p_top = _out[1][0,-1]
          x_top = _out[0][-1]
          # ''' Top pressure and pressure grad check'''
          # q_top = _out[1][:,-1]
          # dqdx = np.linalg.solve(self.A(*q_top, j0), self.F(q_top)).flatten()
          # dpdx = dqdx[0]
          # # Tolerable distance-to-zero-pressure
          # p_min = 0.001e6 # 1 mbar
          # dx_min = 1.0    # 1 m until zero pressure
          # is_reached_vacuum = p_top < p_min or p_top/np.abs(dpdx) < dx_min
          
          # Search criterion
          if x_top >= self.x_mesh[-1]:
            # Register k
            k_last = k
            # break
        if k_last is None:
          raise Exception("Could not bracket lower pressure limit.")

      ''' Sample function p_vent(p_chamber) to find highest-pressure choke. '''
      # Companion function: p_vent(p_chamber)
      search_p = np.linspace(p_minbound, 300e6, 50)
      def penalized_top_pressure(p_chamber):
        soln = self.solve_ssIVP(p_chamber, j0_u(p_chamber))
        if soln[0][-1] < self.x_mesh[-1]:
          return -1
        return soln[1][0, -1]
      tentatives_p_vent = [penalized_top_pressure(p) for p in search_p] 
      # (debug) plot p_chamber to p_vent mapping
      # plt.semilogy(search_p, tentatives_p_vent)
      # plt.xlabel("Inlet pressure (Pa)")
      # plt.ylabel("Vent pressure (Pa)")
      _i = len(tentatives_p_vent) - np.argmax(
        np.array(tentatives_p_vent[::-1]) < 0)

      def bisect_pmin(a,b):
        ''' Manual bisection for minimum pressure. '''
        fn_x_top = lambda p: self.solve_ssIVP(p, j0_u(p))[0][-1]
        # Reject if continuous solution at low bracketing pressure
        if fn_x_top(a) >= self.x_mesh[-1]:
          raise ValueError(f"Pressure {a} is a continuous solution.")
        # Reject if no continuous solution at high bracketing pressure
        if fn_x_top(b) < self.x_mesh[-1]:
          raise ValueError(f"Pressure {b} is not a continuous solution.")

        m = 0.5*(a+b)

        while b - a > brent_atol:
          # Search criterion
          if fn_x_top(m) >= self.x_mesh[-1]: # Continuous solution found
            b = m
          else:
            a = m
          m = 0.5*(a+b)
        return b # Continuous solution

      # Compute minimum possible chamber pressure
      # pc_min = bisect_pmin(p0*2**k/2, p0*2**k)
      pc_min = bisect_pmin(search_p[_i-1], search_p[_i])

      # Compute corresponding minimum vent pressure
      p_vent_min = self.solve_ssIVP(pc_min, j0_u(pc_min))[1][0,-1]
      self._check = (pc_min, j0_u(pc_min))
      print(f"Minimum vent pressure is {p_vent_min}.")
      if p_vent <= p_vent_min:
        print("Choked flow.")
        z = pc_min
        x, (p_soln, h_soln, y_soln, yF_soln), (soln, _) = solve_kernel(z)
      else:
        # Unchoked
        print("Pressure matching for vent pressure at given velocity.")
        # Define wrapped objective taking into subpressurized flow
        def objective(z):
          soln = solve_kernel(z)
          # Retrieve 
          z_top = soln[0][-1]
          p_top = soln[1][0,-1]
          return p_top - p_vent if z_top >= self.x_mesh[-1] else -p_vent
        z = scipy.optimize.brentq(objective, pc_min, z_max, xtol=brent_atol)
        x, (p_soln, h_soln, y_soln, yF_soln), (soln, _) = solve_kernel(z)

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

      return x, (p_soln, h_soln, y_soln, yF_soln), calc_details
    else:
      return x, (p_soln, h_soln, y_soln, yF_soln)


if __name__ == "__main__":
  ''' Perform unit test '''

  ss = SteadyState(np.linspace(-3000,0,200))
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