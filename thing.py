import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import eigh_tridiagonal
from tqdm import tqdm

# Physical constants and parameters
hbar = 1.0
m = 1.0

# Spatial grid
N = 500
x_max = 5.0
x = np.linspace(-x_max, x_max, N)
dx = x[1] - x[0]


#Potentials
def create_harmonic_potential(m, freq, x):
    return  0.5 * m * freq**2 * x**2

def create_inf_square_well(x, width, V0=1e6):
    """
    Create a square well potential:
    V = 0 inside the well (|x| < width/2)
    V = V0 outside the well (|x| > width/2)
    """
    V = np.full_like(x, V0)
    V[np.abs(x) < width/2] = 0
    return V

#Hamiltonian
def create_hamiltonian(m, potential):
    """Create the Hamiltonian matrix for a 1D particle with a given potential."""
    # Kinetic energy term (tridiagonal representation)
    kinetic_prefactor = -hbar**2 / (2 * m * dx**2)
    diagonal = np.full(N, -2.0)
    off_diagonal = np.ones(N - 1)
    T_diag = kinetic_prefactor * diagonal
    T_offdiag = kinetic_prefactor * off_diagonal

    # Full Hamiltonian diagonal is T_diag + V, off-diagonal is T_offdiag
    V = potential
    H_diag = T_diag + V
    H_offdiag = T_offdiag

    return H_diag, H_offdiag

def H_linear(m, v0, v1, start_time, stop_time):
    """
    Return a function H(t) that linearly morphs from potential v0→v1
    over [start_time, stop_time].
    
    m           : mass
    v0, v1      : 1D arrays of length N_x (initial & final V(x))
    start_time  : float
    stop_time   : float
    """
    def H_func(t):
        # fraction through the ramp, clipped to [0,1]
        s = np.clip((t - start_time)/(stop_time - start_time), 0.0, 1.0)
        Vt = v0*(1 - s) + v1*s
        return create_hamiltonian(m, Vt)
    return H_func

def H_infwell(m, x, L0, L1, Vbarrier0, Vbarrier1, start_time, stop_time):
    """
    Return a function H(t) for an infinite‐well whose width expands
      L0 → L1 and barrier height changes Vbarrier0 → Vbarrier1 over the ramp.
    
    m           : mass
    x           : 1D spatial grid
    L0, L1      : initial & final well widths
    Vbarrier0   : initial barrier height (potential outside well)
    Vbarrier1   : final barrier height (potential outside well)
    start_time  : float
    stop_time   : float
    """
    def H_func(t):
        s = np.clip((t - start_time)/(stop_time - start_time), 0.0, 1.0)
        L = L0 + s*(L1 - L0)  # Linear interpolation of width
        Vbarrier = Vbarrier0 + s*(Vbarrier1 - Vbarrier0)  # Linear interpolation of barrier height
        Vt = create_inf_square_well(x, width=L, V0=Vbarrier)
        return create_hamiltonian(m, Vt)
    return H_func


def find_wavefunction(x, H):
    """Find the wavefunction from Hamiltonian."""
    H_diag, H_offdiag = H
    eigvals, eigvecs = eigh_tridiagonal(H_diag, H_offdiag)

    # Normalize eigenstates
    psi = np.zeros((N, len(eigvecs)))
    for j in range(len(eigvecs)):
        
        psi[:, j] = eigvecs[:, j]
        # normalize that column
        norm = np.sqrt(np.trapezoid(np.abs(psi[:, j])**2, x))
        psi[:, j] /= norm

    return eigvals, psi


def freq_func(time, time_start, time_stop, freq0, freq1, sudden_or_adiabat):
    """Frequency function for the time-dependent harmonic oscillator.
    sudden_or_adiabat: 0 for sudden, 1 for adiabatic."""

    
    
    

    if sudden_or_adiabat == 0:
        return np.where(time <time_start, freq0, freq1)
    else:
        ramp_duration = time_stop - time_start
        freq = np.zeros_like(time)
        freq[time < time_start] = freq0
        mask_ramp = (time >= time_start) & (time <= time_stop)
        freq[mask_ramp] = freq0 + (freq1 - freq0) * ((time[mask_ramp] - time_start) / ramp_duration)
        freq[time > time_stop] = freq1

        return freq




def sudden_evolution(time, x, m, v0, v1, n, start_time=0):
    H0, H1 = create_hamiltonian(m, v0), create_hamiltonian(m, v1)
    energy_init, psi_init = find_wavefunction(x, H0)
    energy_final, psi_1 = find_wavefunction(x, H1)

    # Set intial state at nth eigenstate of original Hamiltonian
    psi_init_n = psi_init[:, n]

    #Find Constants
    integrand = np.conj(psi_1) * psi_init_n[:, None]
    c_m = np.trapezoid(integrand, x, axis=0)

    dt = time - start_time
    dt_clamped = np.where(dt > 0, dt, 0.0)
    phase = np.exp(-1j * np.outer(dt_clamped, energy_final) / hbar)
    psi_xt = np.sum(
        c_m[None, None, :]
        * phase[:, None, :]
        * psi_1[None, :, :],
        axis=2
    )  
    return psi_xt, c_m
    

def dyson_evolution(time, x, m, H_series, n, M=5):
    """
    Adiabatic expansion under H(t) = H0 + (H1-H0)*s(t), with global phase removed.
    Returns:
      psi_xt : (N_t, N_x) complex wave‐packet at each time
      c      : (N_t, M) coefficients in instantaneous basis
    """
    N_t = len(time)
    dx = x[1] - x[0]

    # 1) Build instantaneous eigenbases & energies (truncated to M levels)
    E_list, psi_list = [], []
    for t in tqdm(time, desc="Solving Schrödinger Eqn", unit="steps"):
        if callable(H_series):
            Ht = H_series(t)
        else:
            Ht = H_series.pop(0)
        E_vals, psi = find_wavefunction(x, Ht)
        E_list.append(E_vals[:M])
        psi_list.append(psi[:, :M])
    E = np.array(E_list)  # shape (N_t, M)

    # Grab the initial eigenenergy for phase stripping
    E0 = E[0, n]

    # 2) Allocate arrays
    psi_xt = np.zeros((N_t, len(x)), dtype=complex)
    c      = np.zeros((N_t, M), dtype=complex)

    # 3) Initialize in nth eigenstate of H(0)
    c[0, :]      = 0
    c[0,   n]    = 1.0
    psi_xt[0]    = psi_list[0][:, n]  # no phase at t=0

    # 4) Propagate stepwise, then strip the global e^{-iE0 t/ħ} phase
    for i in range(1, N_t):
        dt         = time[i] - time[i-1]
        E_prev     = E[i-1]         # (M,)
        psi_prev   = psi_list[i-1]  # (N_x, M)
        psi_curr   = psi_list[i]    # (N_x, M)

        # (a) dynamic phase under old basis
        c_mid = c[i-1] * np.exp(-1j * E_prev * dt / hbar)

        # (b) basis‐rotation overlap S_{jk} = ⟨ψ_curr_j|ψ_prev_k⟩
        S = psi_curr.conj().T @ psi_prev * dx

        # (c) project onto new instantaneous basis
        c[i] = S @ c_mid

        # (d) rebuild ψ(x) and strip global phase e^{-iE0 t/ħ}
        psi_xt[i] = psi_curr @ c[i]
        psi_xt[i] *= np.exp(+1j * E0 * time[i] / hbar)

    return psi_xt, c


def adiabatic_theorem_evolution(time, x, m, H_series, n):
    """
    Compute psi(x,t) under the adiabatic theorem:
      psi(x,t_i) = psi_n(x; t_i) * exp[-(i/ħ) ∫₀^{t_i} E_n(t') dt'].

    time       : array of shape (N_t,)
    x          : spatial grid of shape (N_x,)
    m          : mass (only used if H_series is callable)
    H_series   : either
                   - a callable H_series(t) -> (H_diag, H_offdiag), or
                   - a list of (H_diag, H_offdiag) of length N_t
    n          : which eigenstate index to track (0-based)
    Returns:
       psi_xt   : complex array (N_t, N_x)
    """
    N_t = len(time)
    N_x = len(x)

    # 1) compute instantaneous psi_n and E_n at each time
    psi_n_list = np.zeros((N_t, N_x), dtype=complex)
    E_n_list   = np.zeros(N_t)
    for i, t in enumerate(time):
        if callable(H_series):
            H_diag, H_off = H_series(t)
        else:
            H_diag, H_off = H_series.pop(0)
        E_vals, psi_all = find_wavefunction(x, (H_diag, H_off))
        psi_n_list[i] = psi_all[:, n]
        E_n_list[i]   = E_vals[n]

    # 2) build cumulative integral of E_n(t) dt via trapezoid
    #    cumtrapz returns length N_t-1, so prepend a zero
    phi_dyn = np.concatenate((
        [0.0],
        -1j/ hbar * cumulative_trapezoid(E_n_list, time)
    ))  # shape (N_t,)

    # 3) assemble psi(x,t)
    psi_xt = psi_n_list * np.exp(phi_dyn)[:, None]


    M   = N_x
    c_m = np.zeros((N_t, M), dtype=complex)
    c_m[:, n] = 1.0+0j

    return psi_xt, c_m


# N_t = 500
# time_length = 4 #seconds
# test_time = np.linspace(0, time_length, N_t)

# v0, v1 = create_inf_square_well(x, 5, 100), create_harmonic_potential(m, 30, x)
# n = 0
# start_time = .5
# stop_time = 1.5
# psi_xt, c_m = sudden_evolution(test_time, x, m, v0, v1, n, start_time=start_time)

# H_well = H_linear(m, v0, v1, start_time, stop_time)
# Hs = H_well
# Hs = H_linear(m, v0, v1, start_time, stop_time)

# psi_adiabatic, c_adiabatic = adiabatic_theorem_evolution(test_time, x, m, Hs, n)




def main(model_type, n, v0_param, v1_param, time_param, N_t = 500, time_length = 4):


    N_t = N_t
    time_length = time_length #seconds
    time = np.linspace(0, time_length, N_t)






    if model_type == 'sudden': #time_param = start_time, end time
        time_param = time_param[0]
        if v0_param[0] == 'Square Well' and v1_param[0] == 'Square Well':
            width0, depth0 = v0_param[1]
            width1, depth1 = v1_param[1]
            v0, v1 = create_inf_square_well(x, width0, depth0), create_inf_square_well(x, width1, depth1)
        elif v0_param[0] == 'Square Well' and v1_param[0] == 'Harmonic':
            width0, depth0 = v0_param[1]
            freq1 = v1_param[1]
            v0, v1 = create_inf_square_well(x, width0, depth0), create_harmonic_potential(m, freq1, x)
        elif v0_param[0] == 'Harmonic' and v1_param[0] == 'Square Well':
            freq0 = v0_param[1]
            width1, depth1 = v1_param[1]
            v0, v1 = create_harmonic_potential(m, freq0, x), create_inf_square_well(x, width1, depth1)
        elif v0_param[0] == 'Harmonic' and v1_param[0] == 'Harmonic':
            freq0 = v0_param[1]
            freq1 = v1_param[1]
            v0, v1 = create_harmonic_potential(m, freq0, x), create_harmonic_potential(m, freq1, x)
        else:
            raise ValueError("Invalid potential types")
        psi_xt, c_m = sudden_evolution(time, x, m, v0, v1, n, start_time=time_param)
        c_m = np.broadcast_to(c_m, (N_t, c_m.shape[0]))
        potential = np.where(
            time[:, None] < time_param,  # shape (N_t,1)
            v0[None, :],                      # broadcast to (N_t,N)
            v1[None, :]                       # broadcast to (N_t,N)
        )   
        

    else: #time_param = (start_time, stop_time)
        start_time, stop_time = time_param
        if v0_param[0] == 'Square Well' and v1_param[0] == 'Square Well':
            width0, depth0 = v0_param[1]
            width1, depth1 = v1_param[1]
            Hs = H_infwell(m, x, width0, width1, depth0, depth1, start_time, stop_time)
        elif v0_param[0] == 'Square Well' and v1_param[0] == 'Harmonic':
            width0, depth0 = v0_param[1]
            freq1 = v1_param[1]
            v0, v1 = create_inf_square_well(x, width0, depth0), create_harmonic_potential(m, freq1, x)
            Hs = H_linear(m, v0, v1, start_time, stop_time)
        elif v0_param[0] == 'Harmonic' and v1_param[0] == 'Square Well':
            freq0 = v0_param[1]
            width1, depth1 = v1_param[1]
            v0, v1 = create_harmonic_potential(m, freq0, x), create_inf_square_well(x, width1, depth1)
            Hs = H_linear(m, v0, v1, start_time, stop_time)
        elif v0_param[0] == 'Harmonic' and v1_param[0] == 'Harmonic':
            freq0 = v0_param[1]
            freq1 = v1_param[1]
            v0, v1 = create_harmonic_potential(m, freq0, x), create_harmonic_potential(m, freq1, x)
            Hs = H_linear(m, v0, v1, start_time, stop_time)
        else:
            raise ValueError("Invalid potential types")

        if model_type == 'dyson':
            psi_xt, c_m = dyson_evolution(time, x, m, Hs, n, M=5)
        elif model_type == 'adiabatic':
            psi_xt, c_m = adiabatic_theorem_evolution(time, x, m, Hs, n)
        else:
            raise ValueError("Invalid model type")


        potential = np.zeros((N_t, N))
        T_diag, _ = create_hamiltonian(m, np.zeros_like(x))
        for i, t in enumerate(time):
            H_d, H_off = Hs(t)        # get (H_diag,H_offdiag) at time t
            potential[i] = H_d - T_diag

        
    return psi_xt, c_m, potential
    
    







def animate_probability_density(psi_xt, x, time):
    print(2)
    fig, ax = plt.subplots()
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, np.max(np.abs(psi_xt)**2)*1.1)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|\psi(x,t)|^2$")
    ax.set_title("Sudden‐Quench Wavepacket Evolution")

    line, = ax.plot([], [], lw=2)

    time_text = ax.text(0.05, 0.95, "", transform=ax.transAxes,
                    fontsize=12, verticalalignment='top')

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame):
        y = np.abs(psi_xt[frame])**2
        line.set_data(x, y)
        ax.set_title(f"t = {time[frame]:.2f}")
        time_text.set_text(f"t = {time[frame]:.2f} s")
        return line, time_text

    ani = FuncAnimation(
        fig, update, frames=len(time),
        init_func=init, blit=True,
        interval=20        # milliseconds between frames
    )

    plt.show()

def animate_wavefunction(psi_xt, x, time):
    """
    Animate the real and imaginary parts of ψ(x,t).

    psi_xt: complex array of shape (N_t, N_x)
    x:      1D array of length N_x
    times:  1D array of length N_t
    """
    # precompute global y-limits
    max_amp = np.max(np.abs(psi_xt))
    y_lim = max_amp * 1.1

    # set up figure + axes
    fig, (ax_re, ax_im) = plt.subplots(2, 1, sharex=True, figsize=(6, 5))
    ax_re.set_xlim(x.min(), x.max())
    ax_re.set_ylim(-y_lim, y_lim)
    ax_re.set_ylabel("Re ψ(x,t)")
    ax_re.grid(True)

    ax_im.set_xlim(x.min(), x.max())
    ax_im.set_ylim(-y_lim, y_lim)
    ax_im.set_ylabel("Im ψ(x,t)")
    ax_im.set_xlabel("x")
    ax_im.grid(True)

    # two line artists
    line_re, = ax_re.plot([], [], lw=2, label="Re")
    line_im, = ax_im.plot([], [], lw=2, label="Im", color="C1")

    # a time‐stamp in the top subplot
    time_text = ax_re.text(
        0.02, 0.90, "", transform=ax_re.transAxes,
        fontsize=12, verticalalignment="top"
    )

    def init():
        line_re.set_data([], [])
        line_im.set_data([], [])
        time_text.set_text("")
        return line_re, line_im, time_text

    def update(frame):
        y_re = np.real(psi_xt[frame])
        y_im = np.imag(psi_xt[frame])
        line_re.set_data(x, y_re)
        line_im.set_data(x, y_im)

        t = time[frame]
        time_text.set_text(f"t = {t:.2f} s")
        return line_re, line_im, time_text

    ani = FuncAnimation(
        fig, update,
        frames=len(time),
        init_func=init,
        blit=True,
        interval=50  # ms between frames; tweak to taste
    )

    plt.tight_layout()
    plt.show()

def animate_all(psi_xt, x, time, rate = 30):
    """
    Animate simultaneously:
      1) |ψ(x,t)|^2
      2) Re ψ(x,t)
      3) Im ψ(x,t)

    psi_xt : complex array, shape (N_t, N_x)
    x      : 1D array of length N_x
    times  : 1D array of length N_t
    """
    # Precompute y-limits
    prob_max = np.max(np.abs(psi_xt)**2) * 1.1
    amp_max  = np.max(np.abs(psi_xt))   * 1.1

    # Set up figure + 3 subplots
    fig, (ax_p, ax_re, ax_im) = plt.subplots(
        3, 1, sharex=True, figsize=(6, 9)
    )

    # ---- configure each axis ----
    ax_p.set_ylabel(r"$|\psi|^2$")
    ax_p.set_xlim(x.min(), x.max())
    ax_p.set_ylim(0, prob_max)
    ax_p.grid(True)

    ax_re.set_ylabel("Re ψ")
    ax_re.set_ylim(-amp_max, amp_max)
    ax_re.grid(True)

    ax_im.set_ylabel("Im ψ")
    ax_im.set_xlabel("x")
    ax_im.set_ylim(-amp_max, amp_max)
    ax_im.grid(True)

    # Create line objects (initially empty)
    line_p, = ax_p.plot([], [], lw=2)
    line_re, = ax_re.plot([], [], lw=2, color="C1")
    line_im, = ax_im.plot([], [], lw=2, color="C2")

    # A single time‐stamp in the top panel
    time_text = ax_p.text(
        0.02, 0.90, "", transform=ax_p.transAxes,
        fontsize=12, verticalalignment="top"
    )

    def init():
        # clear all lines and timestamp
        line_p.set_data([], [])
        line_re.set_data([], [])
        line_im.set_data([], [])
        time_text.set_text("")
        return line_p, line_re, line_im, time_text

    def update(frame):
        t = time[frame]
        psi = psi_xt[frame]            # shape (N_x,)

        # update data for each line
        line_p.set_data(x, np.abs(psi)**2)
        line_re.set_data(x, np.real(psi))
        line_im.set_data(x, np.imag(psi))

        # update the timestamp
        time_text.set_text(f"t = {t:.2f} s")
        return line_p, line_re, line_im, time_text

    ani = FuncAnimation(
        fig, update,
        frames=len(time),
        init_func=init,
        blit=True,
        interval=rate  # ms between frames
    )

    plt.tight_layout()
    plt.show()

def animate_with_populations_sudden(psi_xt, x, times, c_m, V, rate = 50):
    """
    Animate:
      1) |ψ(x,t)|^2
      2) Re ψ(x,t)
      3) Im ψ(x,t)
      4) bar chart of P_m = |c_m|^2 (populations)
    """
    # populations (static for sudden)
    Pm = np.abs(c_m)**2

    # y‐limits
    prob_max = np.max(np.abs(psi_xt)**2)*1.1
    amp_max  = np.max(np.abs(psi_xt))  *1.1
    V_max   = np.max(V) * 1.1

    # make 4 rows
    fig, (ax_p, ax_re, ax_im, ax_bar) = plt.subplots(4,1, figsize=(6,10))

    # 1) probability density
    ax_p.set_ylabel(r"$|\psi|^2$")
    ax_p.set_xlim(x.min(), x.max())
    ax_p.set_ylim(0, prob_max)
    line_p, = ax_p.plot([], [], lw=2)
    
    ax_v = ax_p.twinx()  # create a twin y-axis for the potential
    ax_v.set_yticks([])        # no tick marks
    ax_v.set_ylabel("")        # no label
    ax_v.spines['right'].set_visible(False)
    ax_v.set_ylim(0, V_max)
    line_v, = ax_v.plot([], [], linestyle='--', color = "black")

    # 2) real part
    ax_re.set_ylabel("Re ψ")
    ax_re.set_ylim(-amp_max, amp_max)
    ax_re.set_xlim(x.min(), x.max())
    line_re, = ax_re.plot([], [], lw=2, color="C1")
    ax_re.grid(True)

    # 3) imaginary part
    ax_im.set_ylabel("Im ψ")
    ax_im.set_ylim(-amp_max, amp_max)
    ax_im.set_xlim(x.min(), x.max())
    line_im, = ax_im.plot([], [], lw=2, color="C2")
    ax_im.grid(True)

    # 4) populations bar chart
    ax_bar.set_ylabel(r"$P_m$")
    ax_bar.set_xlabel("m")
    bars = ax_bar.bar(np.arange(len(Pm)), Pm, color="C3")
    ax_bar.set_xlim(0-.5, 5+.5) 
    ax_bar.set_ylim(0, 1.0)  # populations sum to 1 or less
    ax_bar.grid(True)

    # shared timestamp in top panel
    time_text = ax_p.text(0.02, 0.90, "", transform=ax_p.transAxes,
                          fontsize=12, verticalalignment="top")

    def init():
        for artist in (line_p, line_v, line_re, line_im):
            artist.set_data([], [])
        time_text.set_text("")
        # bars are already set to Pm
        return (line_p, line_v, line_re, line_im, time_text, *bars)

    def update(frame):
        psi = psi_xt[frame]
        t   = times[frame]
        

        # update wave plots
        line_p.set_data(x, np.abs(psi)**2)
        line_re.set_data(x, np.real(psi))
        line_im.set_data(x, np.imag(psi))
        
        V_frame = V[frame]  # shape (N,)
        Vmax_now = V_frame.max() * 1.1
        ax_v.set_ylim(0, Vmax_now)
        line_v.set_data(x, V_frame)

        # for adiabatic: if you have c_m(t), you'd do:
        #    new_Pm = np.abs(c_m_t[frame])**2
        #    for bar, h in zip(bars, new_Pm):
        #        bar.set_height(h)

        # timestamp
        time_text.set_text(f"t = {t:.2f} s")

        return (line_p, line_v, line_re, line_im, time_text, *bars)

    ani = FuncAnimation(
        fig, update,
        frames=len(times),
        init_func=init,
        blit=True,
        interval=rate #ms
    )

    plt.tight_layout()
    plt.show()

def animate_with_populations_dyson(psi_xt, x, times, c_m, V, rate=50):
    """
    Animate:
      1) |ψ(x,t)|^2
      2) Re ψ(x,t)
      3) Im ψ(x,t)
      4) bar chart of P_m(t) = |c_m(t)|^2 (instantaneous populations)
    psi_xt : (N_t, N_x) array of wavefunctions
    c_m    : (N_t, M) array of expansion coefficients
    V      : (N_t, N_x) or (N_x,) potential(s)
    """
    N_t, N_x = psi_xt.shape
    M = c_m.shape[1]
    P0 = np.abs(c_m[0])**2

    # precompute y‐limits
    prob_max = np.max(np.abs(psi_xt)**2) * 1.1
    amp_max  = np.max(np.abs(psi_xt))       * 1.1
    V_max    = np.max(V if V.ndim>1 else V) * 1.1

    # set up figure + axes
    fig, (ax_p, ax_re, ax_im, ax_bar) = plt.subplots(4,1, figsize=(6,10))
    # probability
    ax_p.set_ylabel(r"$|\psi|^2$")
    ax_p.set_xlim(x.min(), x.max())
    ax_p.set_ylim(0, prob_max)
    ax_p.grid(True)
    line_p, = ax_p.plot([], [], lw=2)
    # potential overlay
    ax_v = ax_p.twinx()
    #ax_v.set_yticks([]); #ax_v.spines['right'].set_visible(True)
    ax_v.set_ylim(0, V_max)
    line_v, = ax_v.plot([], [], '--', color='k')

    # real part
    ax_re.set_ylabel("Re ψ"); ax_re.set_xlim(x.min(), x.max())
    ax_re.set_ylim(-amp_max, amp_max); ax_re.grid(True)
    line_re, = ax_re.plot([], [], lw=2, color="C1")

    # imag part
    ax_im.set_ylabel("Im ψ"); ax_im.set_xlim(x.min(), x.max())
    ax_im.set_ylim(-amp_max, amp_max); ax_im.grid(True)
    line_im, = ax_im.plot([], [], lw=2, color="C2")

    # populations bar chart
    ax_bar.set_ylabel(r"$P_m$")
    ax_bar.set_xlabel("m")
    bars = ax_bar.bar(np.arange(M), P0, color="C3")
    ax_bar.set_xlim(-0.5, M-0.5)
    ax_bar.set_ylim(0, 1.0)
    ax_bar.grid(True)

    # timestamp
    time_text = ax_p.text(0.02, 0.90, "", transform=ax_p.transAxes,
                          fontsize=12, verticalalignment="top")

    def init():
        for art in (line_p, line_v, line_re, line_im):
            art.set_data([], [])
        for bar in bars:
            bar.set_height(0)
        time_text.set_text("")
        return (line_p, line_v, line_re, line_im, time_text, *bars)

    def update(i):
        psi = psi_xt[i]
        t   = times[i]
        # update wave plots
        line_p.set_data(x, np.abs(psi)**2)
        line_re.set_data(x, np.real(psi))
        line_im.set_data(x, np.imag(psi))

        # update potential
        Vframe = V[i] if V.ndim>1 else V
        ax_v.set_ylim(0, Vframe.max()*1.1)
        ax_v.set_yticks(Vframe[::int(N_x/5)])
        line_v.set_data(x, Vframe)

        # update populations
        Pm = np.abs(c_m[i])**2
        for bar, h in zip(bars, Pm):
            bar.set_height(h)

        # timestamp
        time_text.set_text(f"t = {t:.2f} s")

        return (line_p, line_v, line_re, line_im, time_text, *bars)

    ani = FuncAnimation(fig, update, frames=N_t,
                        init_func=init, blit=True,
                        interval=rate)
    plt.tight_layout()
    plt.show()




