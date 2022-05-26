# Functional Linear Regression of CDFs

This repository is the official implementation of the numerical experiments in the paper "Functional Linear Regression of CDFs". 

## Requirements

Python version we used: Python 3.6.9.

R version we used: R version 4.1.2.

To install requirements for the simulation studies in Section 4.1:

```
pip3 install -r requirements.txt
```

To install R packages used in the real data experiment in Section 4.2:

```{r}
# Run the following R codes in R.
install.packages('hdm')
install.packages('stats')
install.packages('quadprog')
```


## Simulation studies (Section 4.1)

### Estimation errors against sample size (Figure 1a)

To obtain the $\ell^2$ errors and KS distances under different sample sizes with the simulated sample and generate Figure 1a, run this command:

```
python3 simulation.py
```

### Estimation errors against dimension (Figure 1b)

To obtain the $\ell^2$ errors and KS distances under different dimensions with the simulated sample and generate Figure 1b, run this command:

```
python3 sim_dim.py
```


## Real data experiments (Section 4.2)

### Estimation errors of the estimator and the projected estimator (Figure 1c)

To obtain the $\ell^2$ errors of the estimator and the projected estimator in the real data experiments, run:

```
Rscript exp_real_data.R
```

To plot Figure 1c, run

```
python3 plot_real_data.py
```


## Results

After running the above commands, our codes 

- save the estimation errors under different sample sizes in the simulation studies in 'sample_size.npz',

- generate Figure 1a in 'sim_sample_size.pdf', 

- save the estimation errors under different dimensions in the simulation studies in 'dimension.npz',

- generate Figure 1b in 'sim_dimension.pdf',

- save the $\ell^2$ errors of the estimator and the projected estimator in the real data experiments in 'l2error_mat.txt' and 'l2error_proj_mat.txt' respectively,

- generate Figure 1c in 'exp_real_data.pdf'.



