# Market Regime Clustering

## Description

Market Regime Clustering using Wasserstein K-Means (Hovarth, Issa, Miguruza, 2021). This project implements original univariate clustering based on Wasserstein distance. It contributes to the original paper in a following way:



* Multivariate clustering
* Comparison of results with statistical breakpoint tests
* Trading strategy based on regime changes







## Testing

We test the validity of breaks in proposed by the clusterization in two ways:

1. Using Chow test (for levels), Levene test (for variance) and Kolmogorov-Smirnov test to check whether the distribution changes
2. Comparing the results with the QLR test for unknown-date breakpoint detection (Andrews, 1993; Bai and Perron, 2003)

## 

## Bibliography

1. Horvath, B., Issa, Z., \& Muguruza, A. (2021). Clustering market regimes using the Wasserstein distance. arXiv:2110.11848 \[q-fin.CP]. https://arxiv.org/abs/2110.11848
2. Bai, J., and Perron, P. (2003). Computation and analysis of multiple structural change models. Journal of Applied Econometrics, 18(1), 1–22.
3. Donald W. K. Andrews (1993). Tests for parameter instability and structural change with unknown change point. Econometrica, 61(4), 821–856

