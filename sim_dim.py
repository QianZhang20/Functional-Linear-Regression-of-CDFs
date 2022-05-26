# This is the code for the simulation study with respect to dimension. 
# It will generate Figure 1b in the paper.

## Import required packages
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from qpsolvers import solve_qp
## Enable the use of Latex in the texts of the figure.
mpl.rcParams['text.usetex'] = True

## The function to calculate the matrix $U_n$ defined in the paper.
def Phi2_integral(x, r_seq, d):
    p2 = np.zeros((d, d))
    val = np.mean(1/x)
    
    for i1 in range(d):
        r = r_seq[i1]
        for i2 in range(d):
            s = r_seq[i2]
            
            p2[i1, i2] = 2 + (1/(r+s+1)-1) * val
            
    return p2 / 2

## The function to calculate the vector $\sum_{j=1}^n \int_S I_{y^{(j)}}\Phi_j dm$ 
##   appeared in eq. (3) in the paper.
def Phi_integral(x, y, r_seq, d):
    p1 = np.zeros(d)
    val1 = np.mean(1/x)
    for i in range(d):
        r = r_seq[i]
        val2 = np.mean((x**r)*(y**(r+1)))
        p1[i] = 2 + (1/(r+1)-1)*val1 - val2/(r+1)
    return p1 / 2

## The function to calculate the estimated parameter using eq. (3) in the paper.
def L2_penalized_unconstrained(x, y, lam, r_seq, d):
    lam_n = lam / np.size(x)
    A = Phi2_integral(x, r_seq, d) + np.diag(np.ones(d)*lam_n)
    b = Phi_integral(x, y, r_seq, d)
    res = np.matmul(np.linalg.inv(A), b)
    return res

## The function to approximately calcualate the KS distance between the estimated 
##    CDF and the true CDF. 
## It will use equally spaced grid points to estimate KS distance.
def KS_statistic(dtheta, t_mat):
    dF_seq = np.abs(np.matmul(t_mat, dtheta))
    return np.max(dF_seq)
    

## Pass arguments to the code.
parser = argparse.ArgumentParser(description='Simulation on dimension d')
### Set the regularization parameter $\lambda$ in eq. (2) & (3) of the paper.
parser.add_argument('--lam', default=0.001, type=float, help='lambda')
### Set the sample size for the estimation in this simulation study.
parser.add_argument('--N', default=100000, type=int, help='sample size')
### Set the maximum dimension for the estimation in this simulation study.
parser.add_argument('--d_max', default=100, type=int, help='maximum dim')
### Set the number of independent runs in the simulation.
parser.add_argument('--rep', default=100, type=int, help='number of repetition')
### Set the minimum dimension for the estimation in this simulation study.
parser.add_argument('--d_min', default=10, type=int, help='minimum dim')
### Set the number of dimensions between two neighboring estimations in this simulation study.
parser.add_argument('--gap', default=5, type=int, help='gap between each dimension')
### Set the distance between two neighboring grid points used to estimate KS distance.
parser.add_argument('--error', default=0.0002, type=float, help='error for evaluating KS statistic')

args = parser.parse_args()

## Set the arguments accordingly.
lam = args.lam
gap = args.gap
N = args.N
d_min = args.d_min
d_max = args.d_max
rep = args.rep
error_KS = args.error

## Generate the true parameter theta_true_nonzero of dimension d_min from Dirichlet(1,...,1).
## When the dimension is larger than d_min, the true parameter is the vector obtained by 
##    appending 0 to theta_true_nonzero for the remaining coordinates.
np.random.seed(59)
alpha = np.ones(d_min)
theta_true_nonzero = np.random.dirichlet(alpha, 1)[0,]

## Store the dimensions when an estimation is conducted.
dimension = np.arange(d_min, gap+d_max, gap)
dimension[-1] = d_max
## Store the calculated $\ell^2$ error and KS distance.
l2_error = np.zeros((len(dimension), rep))
KS_error = np.zeros((len(dimension), rep))

## Set the grid points used to estimate KS distance
t_seq = np.arange(0, 1+error_KS, error_KS)

## Set the random seed for this simulation.
np.random.seed(59)

## Iteration of independent runs.
for k in range(rep):
    if k % 1 == 0:
        print(k)
    ind = 0
    ## Iteration of dimensions where an estimation is conducted.
    for d in range(d_min, gap+d_max, gap):
        ## Set the matrices and vectors used to solve the regularized quadratic programming
        ##    question used to calculate the $\ell^2$-projection of the estimated parameter,
        ##    i.e., $\widetilde{\theta}_{\lambda}$ in the paper.
        ## $\widetilde{\theta}_{\lambda}$ is used to calculate the estimated CDF to estimate
        ##    KS distance.
        P_mat = np.diag(np.ones(d))
        A_mat = np.ones(d)
        b_vec = np.array([1.])
        lb = np.zeros(d)

        ## Set the true parameter.
        theta_true = np.zeros(d)
        theta_true[:d_min] = theta_true_nonzero
        
        ## r_seq is the vector of $r(i)$ in eq. (16) of the paper.
        r_seq = np.zeros(d)
        for i in range(d):
            if i <= ((d+1)//2-1):
                r_seq[i] = i + 1
            else:
                r_seq[i] = 2 / (2*(i+1)-d+1)
        ## Set the grid points used to estimate the KS distance.
        t_mat = np.zeros((len(t_seq),d))
        for i in range(d):
            t_mat[:,i] = t_seq**r_seq[i]
        
        ## Generate the simulated sample of size N.
        x = np.random.uniform(0.5, 2, N)
        y = np.zeros(N)
        for j in range(N):
            i = np.random.choice(d, 1, p=theta_true)[0]
            q = np.random.random(1)[0]
            y[j] = q**(1/r_seq[i]) / x[j]
        
        ## Estimate the parameter.
        theta_est = L2_penalized_unconstrained(x, y, lam, r_seq, d)
        ## Calculate $\ell^2$ error.
        l2_error[ind, k] = np.linalg.norm(theta_est - theta_true)
        ## Calculate the projected parameter.
        theta_proj = solve_qp(P=P_mat, q=-theta_est, A=A_mat, b=b_vec, lb=lb)
        ## Calculate the KS distance.
        KS_error[ind, k] = np.max(np.abs(np.matmul(t_mat, theta_proj - theta_true)))
        ind = ind+1

## Save the errors.
np.savez("dimension", l2_error, KS_error, dimension)


## Take log scale.
dim_log = np.log(dimension)
l2_error_d_log = np.log(l2_error)
KS_error_d_log = np.log(KS_error)

## To store the mean and confidence intervals of the errors with independent runs.
l2_error_d_log_mean = np.mean(l2_error_d_log, 1)
KS_error_d_log_mean = np.mean(KS_error_d_log, 1)
l2_error_d_log_upper = np.copy(l2_error_d_log_mean)
l2_error_d_log_lower = np.copy(l2_error_d_log_mean)
KS_error_d_log_upper = np.copy(KS_error_d_log_mean)
KS_error_d_log_lower = np.copy(KS_error_d_log_mean)

## To calculate the 90% confidence intervals of the errors with independent runs.
quantile = 95
for j in range(len(KS_error_d_log_mean)):
    l2_error_d_log_upper[j] = np.percentile(l2_error_d_log[j,], quantile)
    l2_error_d_log_lower[j] = np.percentile(l2_error_d_log[j,], 100-quantile)
    
    KS_error_d_log_upper[j] = np.percentile(KS_error_d_log[j,], quantile)
    KS_error_d_log_lower[j] = np.percentile(KS_error_d_log[j,], 100-quantile)



## Codes to generate the plot of $\ell^2$ error and KS distance versus dimension in log scale.
plt.figure(figsize = (7,6))
plt.rcParams.update({'figure.autolayout': False})
plt.rcParams.update({'font.size': 20})
plt.fill_between(dim_log, l2_error_d_log_lower, l2_error_d_log_upper, color="blue", alpha=0.3)
plt.plot(dim_log, l2_error_d_log_mean, color="blue", linewidth=2.3, label=r"$\ell^2$ error")
plt.fill_between(dim_log, KS_error_d_log_lower, KS_error_d_log_upper, color="red", alpha=0.3)
plt.plot(dim_log, KS_error_d_log_mean, color="red", linestyle='dashdot', linewidth=2.3, label=r"KS distance")
plt.xticks(np.arange(np.floor(min(dim_log))+0.5, np.ceil(max(dim_log)), 0.5))
plt.xlabel(r"log of dimension ($\log d$)", fontsize=25)
plt.ylabel(r"log of error", fontsize=25)
plt.ylim(top=1.1)
plt.legend(loc='upper center',ncol=2, fontsize=20)
plt.grid()
## Save the plot in "sim_dimension.pdf".
plt.savefig("sim_dimension.pdf")
plt.close()


