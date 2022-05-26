# This file load the $\ell^2$ errors of the real data experiments obtained 
#    by runing 'Rscript exp_real_data.R' to draw Figure 1c. 

## Import packages to load the results and plot.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
## Enable the use of Latex in the texts in the figure.
mpl.rcParams['text.usetex'] = True

## Load the $\ell^2$ errors of the estimator and projected estimator.
l2error = np.loadtxt("l2error_mat.txt")
l2error_proj = np.loadtxt("l2error_proj_mat.txt")

## Calculate the mean of the errors
l2error_mean = np.mean(l2error, 1)
l2error_proj_mean = np.mean(l2error_proj, 1)

## Allocate the vectors to store the lower and upper bounds of the confidence intervals.
l2error_upper = np.copy(l2error_mean)
l2error_lower = np.copy(l2error_mean)
l2error_proj_upper = np.copy(l2error_mean)
l2error_proj_lower = np.copy(l2error_mean)

## Calculate the lower and upper bounds of the 90% confidence intervals.
quantile = 95
for j in range(len(l2error_mean)):
    l2error_upper[j] = np.percentile(l2error[j,], quantile)
    l2error_lower[j] = np.percentile(l2error[j,], 100-quantile)
    
    l2error_proj_upper[j] = np.percentile(l2error_proj[j,], quantile)
    l2error_proj_lower[j] = np.percentile(l2error_proj[j,], 100-quantile)

## Plot the $\ell^2$ errors versus the sample sample (Figure 1c) and store it in 'exp_real_data.pdf'.
plt.figure(figsize = (7,6))
plt.rcParams.update({'figure.autolayout': False})
plt.rcParams.update({'font.size': 20})
plt.fill_between(np.arange(1,4001), l2error_lower[0:4000], l2error_upper[0:4000], color="blue", alpha=0.3)
plt.plot(np.arange(1,4001), l2error_mean[0:4000], linewidth=2.3, color="blue", label=r"$\widehat{\theta}_{\lambda}$")
plt.fill_between(np.arange(1,4001), l2error_proj_lower[0:4000], l2error_proj_upper[0:4000], color="red", alpha=0.3)
plt.plot(np.arange(1,4001), l2error_proj_mean[0:4000], linewidth=2.3, linestyle='dashdot', color='red', label=r"$\widetilde{\theta}_{\lambda}$")
plt.xlabel(r"sample size: $n$", fontsize=25)
plt.ylabel(r"$\ell^2$ error", fontsize=25)
plt.legend(loc='upper center',ncol=2, fontsize=20)
plt.grid()
plt.savefig("exp_real_data.pdf")
plt.close()

