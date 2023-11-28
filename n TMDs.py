import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import a1

"""
Variables for user input:
Line 185: change n
Line 188: cahnge the type of distribution used
Function: manual_TMD_w - cahgne the frequencies list and make sure its length matches n

"""

def MLKF_ndof(n, m, m0, l, l0, k, k0, f0=.25):
    """Return mass, damping, striffness and force matrices for nDOF system"""

    M = M_matrix(m, m0)
    L = LK_matrix(l, l0)
    K = LK_matrix(k, k0)
    F = np.zeros(n+1)
    F[0] = f0

    return M, L, K, F

def normal_TMD_w(mu, sigma, n):
    """
    Generates a list of frequncies that the TMDs should be tuned to
    according to the normal distribution

    """
    a = 1 / (n+1)          ## set the area between each w_t
    p = np.arange(0, 1, a)
    w_t = mu + sigma*norm.ppf(p[1:])

    return w_t

def uniform_TMD_w(mu, sigma, n):
    """
    Generates a list of frequncies that the TMDs should be tuned to
    according to the uniform distribution

    """
    # sigma in this function gives the interval wanted
    spacing_array = sigma*np.arange(1, n+1, 1)
    av = np.sum(spacing_array) / n
    f = np.array([mu/(2*np.pi) + item - av for item in spacing_array])
    w = 2*np.pi * f

    return w

def poisson_TMD_w(mu, n):
    """
    Generates a list of frequncies that the TMDs should be tuned to
    according to the poisson distribution

    """

    return np.random.poisson(mu, n)

def binomial_TMD_w(mu, n):
    """
    Generates a list of frequncies that the TMDs should be tuned to
    according to the binomial distribution

    """
    p = mu / n
    return np.random.binomial(n, p)

def log_TMD_w(mu, sigma, n):
    """
    Generates a list of frequncies that the TMDs should be tuned to
    be linear if plotted a logarithmic scale

    """
    # sigma in this case is the factor by which each tuned frequency is multiplied by to get the next one

    start = mu / sigma ** (n // 2)
    f = []
    for i in range(n):
        f.append(start * sigma**i)

    w = 2*np.pi*np.array(f)

    return w

def manual_TMD_w(mu, n):
    """
    Generates a list of frequncies that the TMDs should be tuned to
    according to manual inputs

    """
    print(mu)
    w_t = 2*np.pi * np.array([3.67])
    assert len(w_t) == n
    #assert np.sum(w_t) / n - mu < 0.01

    return w_t

def get_TMD_masses(k, w):
    """
    Generates the list of masses required to tune each TMD to the chosen
    frequencies with the decided distribution
    """

    m = k / (w**2)

    return m

def LK_matrix(c, c0):
    """
    Generates the Damping and Stiffness matrices
    Requires numpy arrays to be passed to it
    """
    C = np.eye(len(c)+1)
    c_sum = np.sum(c)
    edge = np.append([c0 + c_sum], -c)
    #print(edge)
    for i in range(len(c)):
        C[i+1][i+1] = c[i]
    for i in range(len(edge)):
        C[i][0] = edge[i]
        C[0][i] = edge[i]
    
    return C

def M_matrix(m, m0):
    """
    Generates the Mass matrix
    Requires numpy arrays to be passed to it
    """
    M = np.eye(len(m)+1)
    diag = np.append([m0], m)
    for i in range(len(diag)):
        M[i][i] = diag[i]
    return M    

def freq_response(w_list, M, L, K, F):              ## Copied from a1 for ease of use

    """Return complex frequency response of system"""

    return np.array(
        [np.linalg.solve(-w*w * M + 1j * w * L + K, F) for w in w_list]
    )

def plot_freq(n, w_tuned_distribution, sigma, f_list, f_response):
    ## Titles
    title_uniform = str(n) + " dampers, " + w_tuned_distribution + " " + str(sigma) + "Hz spacing"
    title_normal = str(n) + " dampers, normally distributed with σ = " + str(sigma)
    title_manual = str(n) + " damper, tuned to 3.67"
    if w_tuned_distribution == "uniform":
        title = title_uniform
    elif w_tuned_distribution == "normal":
        title = title_normal
    elif w_tuned_distribution == "manual":
        title = title_manual
    else:
        title = title_manual

    ## Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlabel("Frequency/Hz")
    ax.set_ylabel("Displacement/metre")
    ax.plot(f_list, f_response[:,0])

    folder_path = "C:/Users/nirma/OneDrive - University of Cambridge/Engineering Tripos IB/2CW Labs/Integrated Coursework/Graphs/Frequency responses"
    plt.savefig(f"{folder_path}/{title}.png")
    plt.show()

    return

def plot_time(n, w_tuned_distribution, sigma, t_list, t_response):
    ## Titles
    title_uniform = str(n) + " dampers, " + w_tuned_distribution + " " + str(sigma) + "Hz spacing"
    title_normal = str(n) + " dampers, normally distributed with σ = " + str(sigma)
    title_manual = str(n) + " damper, tuned to 3.67"
    if w_tuned_distribution == "uniform":
        title = title_uniform
    elif w_tuned_distribution == "normal":
        title = title_normal
    elif w_tuned_distribution == "manual":
        title = title_manual
    else:
        title = title_manual

    ## Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlabel("Time/s")
    ax.set_ylabel("Displacement/metre")
    ax.plot(t_list, t_response[:,0])

    folder_path = "C:/Users/nirma/OneDrive - University of Cambridge/Engineering Tripos IB/2CW Labs/Integrated Coursework/Graphs/Time responses"
    plt.savefig(f"{folder_path}/{title}.png")
    plt.show()

    return

## Defining frequency and time  lists
f_start = 0
f_end = 8
w_list = np.linspace(f_start, 2*np.pi*f_end, 10001)
t_start = 0
t_end = 8
t_list = np.linspace(t_start, t_end, 10001)


m0 = 3.94    ## mass of floor - number from the A1 lab
k0 = 2095    ## stiffness of floor to ground - number from the A1 lab
kn = 80      ## constant stiffness of TMDs mounting - number from the A1 lab - the idea is to keep the stiffness constant for each TMD 
l0 = 1.98    ## damping rate of floor to ground - number from the A1 lab
ln = 0.86    ## damping rate of TMDs mounting - number from the A1 lab

half_power_bw = 0.08
w_n = np.sqrt(k0/m0)

n = 10                               ## type in the number of dampers wanted
k = kn*np.ones(n)                   ## deciding the stiffnesses of TMDs (this case is equal stiffness)
l = ln*np.ones(n)                   ## deciding the damping rates of TMDs (this case is equal damping rates)
w_tuned_distribution = "uniform"        ## type in the type of distribution
w_tuned_distribution = w_tuned_distribution.lower()    ## putting it all into lowercase for ease

# Setting standard deviation (comment one of these lines out)
#sigma = half_power_bw/2            ## setting standard deviation to be half of the half power bandwidth
sigma = .5                          ## arbitrary value of sigma
#sigma = 2*np.pi*0.6                 ## chosen to be about 1/2 of the modulation delta_w

# Choosing distribution for spacing out the tuned frequencies for the TMDs
if w_tuned_distribution == "normal":
    w_t = normal_TMD_w(w_n, sigma, n)
elif w_tuned_distribution == "poisson":
    w_t = poisson_TMD_w(w_n, n)
elif w_tuned_distribution == "binomial":
    w_t = binomial_TMD_w(w_n, n)
elif w_tuned_distribution == "uniform":
    w_t = uniform_TMD_w(w_n, sigma, n)
elif w_tuned_distribution == "manual":
    w_t = manual_TMD_w(w_n, n)
else:                                ## defaulting to manual distribution
    w_t = manual_TMD_w(w_n, n)
print(w_t)

# Calculating relevant masses for these tuned freqncies
m = get_TMD_masses(k, w_t)

# Generating the matrices
M, L, K, F = MLKF_ndof(n, m, m0, l, l0, k, k0, f0=.25)
print(M)
mass_sum = np.sum(M) - m0
print(f"Sum of damper masses = {mass_sum}")


# Solving the matrix equations
f_response = np.abs(freq_response(w_list, M, L, K, F))
t_response = a1.time_response(t_list, M, L, K, F)
print(t_response)


# Plotting Results
f_list = w_list/(2*np.pi)

plot_type = "both" # type in f for frequency plotss, t for time plots

if plot_type == "f":
    plot_freq(n, w_tuned_distribution, sigma, f_list, f_response)
elif plot_type == "t":
    plot_time(n, w_tuned_distribution, sigma, t_list, t_response)
# default to plotting both
else:
    plot_freq(n, w_tuned_distribution, sigma, f_list, f_response)
    plot_time(n, w_tuned_distribution, sigma, t_list, t_response)