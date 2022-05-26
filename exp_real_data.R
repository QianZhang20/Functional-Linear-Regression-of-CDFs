# This is the code for the real data experiment. 
# It will generate the data used to draw Figure 1c in the paper.

## 'hdm' contains the dataset.
library(hdm)
## Use the function 'integrate' in 'stats' to calculate numerical integration
library(stats)
## Use the function 'solve.QP' in 'quadprog' to calculate the projected parameter by
##   solving a constrained quadratic programming problem.
library(quadprog)
## load the dataset.
data(pension)

## Obtain sample size of the dataset.
n = length(pension$ira)
## Set the number of covariates used which is also the dimension of the parameter.
d = 10
## Covariate matrix.
X = matrix(0, n, d)

## Copy the data of covariates to the covariate matrix.
X[,1] = c(pension$age)
X[,2] = c(pension$inc)
X[,3] = c(pension$fsize)
X[,4] = c(pension$educ)
X[,5] = c(pension$marr)
X[,6] = c(pension$twoearn)
X[,7] = c(pension$db)
X[,8] = c(pension$pira)
X[,9] = c(pension$hown)
X[,10] = c(pension$p401)

## Standardize the continuous covariates (covariate 1 to 4) to have mean 0 and variance 1.
for(i in 1:4){
  X[,i] = X[,i] - mean(X[,i])
  X[,i] = X[,i] / sqrt(var(X[,i]))
}

## Standardize the response to have mean 0 and variance 1.
Y = pension$net_tfa
Y = Y - mean(Y)
Y = Y / sqrt(var(Y))

## Set the random seed used for the 100 independent runs.
set.seed(825)
## Number of independent runs. 
## In each independent run, the data is split into two parts randomly used for learning the 
##   the bases functions and estimating the parameter.
rep_total = 100
## Matrices to store the $\ell^2$ error of the estimation.
l2_error = matrix(0, n*2/3, rep_total)
l2_error_proj = matrix(0, n*2/3, rep_total)

## Set the regularization parameter $\lambda$ in eq. (2) & (3) of the paper.
lam = 10

## Iteration of independent runs.
for(rep in 1:rep_total){
  ## Randomly permute the dataset.
  ran_ord = sample(1:n, n)
  ## X_basis and Y_basis is used to estimate the basis function $\Phi$.
  ## X_full and Y_full is used to estimate the parameter $\theta_*$ using the learned basis $\Phi$.
  X_basis = X[ran_ord[1:n/3],]
  X_full = X[ran_ord[(n/3+1):n],]
  Y_basis = Y[ran_ord[1:n/3]]
  Y_full = Y[ran_ord[(n/3+1):n]]
  
  ## The total number of sample used to estimate the parameter.
  N = length(Y_full)
  
  ## Using Gaussian CDFs as bases with the mean being a linear function of the covariate.
  beta1 = rep(0, d)
  beta0 = rep(0, d)
  sigma = rep(0, d)
  
  ## Estimate the linear coefficients of the Gaussian linear model for the bases.
  ## Set the variance of the Gaussian CDF to be the variance of the residuals.
  for(i in 1:d){
    beta1[i] = (sum(X_basis[,i]*Y_basis) - mean(X_basis[,i])*sum(Y_basis)) / (sum(X_basis[,i]^2) - sum(X_basis[,i])*mean(X_basis[,i]))
    beta0[i] = mean(Y_basis) - beta1[i]*mean(X_basis[,i])
    sigma[i] = sqrt(var(Y_basis - beta1[i]*X_basis[,i] - beta0[i]))
  }
  
  ## Calculate the means of the Guassian base CDFs of each data point used to estimate the parameter.
  n_full = length(Y_full)
  mu_full = matrix(0, n_full, d)
  for(i in 1:d){
    mu_full[,i] = beta1[i]*X_full[,i] + beta0[i]
  }
  
  ## Store the estimated parameters using eq. (3) in theta_mat.
  ## Store the projected parameters in theta_proj_mat.
  theta_mat = matrix(0, n_full, d)
  theta_proj_mat = matrix(0, n_full, d)
  
  ## Matrices and vectors to formulate the quadratic programming problem used to obtain the projected parameter.
  Dmat <- diag(1,d,d)
  Amat <- t(rbind(rep(1,d),diag(1,d,d)))
  bvec <- c(1, rep(0,d))
  
  ## standard deviation of the Gaussian measure m used in eq. (3).
  meas_sigma = 10
  ## Phi_y store the vector $\sum_{j=1}^n \int_S I_{y^{(j)}}\Phi_j dm$ in eq. (3).
  Phi_y = rep(0, d)
  ## Phi2 store the matrix $U_n$ defined in the paper.
  Phi2 = matrix(0, d, d)
  
  ## Estimate the parameter with sample of increasing size one by one.
  for(j in 1:n_full){
    ## Calculate Phi2 and Phi_y using numerical integration.
    for(i1 in 1:d){
      phi_y_func = function(t){
        return(pnorm(t, mu_full[j, i1], sigma[i1])*dnorm(t,0,meas_sigma))
      }
      Phi_y[i1] = Phi_y[i1] + integrate(phi_y_func, Y_full[j], Inf)$value
      for(i2 in 1:d){
        phi2_func = function(t){
          return(pnorm(t, mu_full[j,i1], sigma[i1])*pnorm(t, mu_full[j,i2], sigma[i2])*dnorm(t,0,meas_sigma))
        }
        Phi2[i1, i2] = Phi2[i1, i2] + integrate(phi2_func, -Inf, Inf)$value
      }
    }
    ## Calculate the estimator $\widehat{theta}_{\lambda}$ using eq. (3).
    theta_mat[j,] = solve(Phi2 + diag(lam,d,d))%*%Phi_y
    ## Calculate the $\ell^2$ projected estimator $\widetilde{theta}_{\lambda}$ using quadratic programming.
    dvec = theta_mat[j,]
    qp = solve.QP(Dmat, dvec, Amat, bvec, meq = 1)
    theta_proj_mat[j,] = qp$solution
  }
  
  ## Print the index of independent run.
  print(rep)
  
  ## Calculate and store the $\ell^2$ error of the estimator and projected estimator.
  for(j in 1:n_full){
    l2_error[j, rep] = sqrt(sum((theta_mat[j,]-theta_mat[n_full,])^2))
    l2_error_proj[j, rep] = sqrt(sum((theta_proj_mat[j,]-theta_proj_mat[n_full,])^2))
  }
  
}

## Save the $\ell^2$ errors used to draw Figure 1c.
write.table(l2_error, file="l2error_mat.txt", row.names=FALSE, col.names = FALSE)
write.table(l2_error_proj, file="l2error_proj_mat.txt", row.names=FALSE, col.names = FALSE)


