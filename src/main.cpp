#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <numeric>
#include <random>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace std;
using namespace Eigen;

/*
 * Logistic regression IRLS implementation.
 * Based on https://bwlewis.github.io/GLM/#svdnewton
 * Perform likelihood ratio test for hypothesis testing
 */

/*
 * The following functions follow the interface for GLM, currently only logistic regression is implemented.
 * However, it is straightforward to extend to other GLM family.
 */

double link(double mu)
{
    return log(mu / (1 - mu));
}
double linkinv(double eta)
{
    return 1 / (1 + exp(-eta));
}

double variance(double g)
{
    return g * (1 - g);
}

double mu_eta(double eta)
{
    // derivative of mu to eta
    return exp(-eta) / pow((1 + exp(-eta)), 2);
}

double logistic_reg(const MatrixXd &X, const VectorXd &y, Eigen::Ref<VectorXd> b, int max_iter, double tol)
{
    // b is the effect sizes to be optimized in place, also used as the starting points
    // return loglikelihood
    int n = X.rows();
    int m = X.cols();
    JacobiSVD<MatrixXd> svd(X, ComputeThinU | ComputeThinV);
    VectorXd eta = X * b;
    VectorXd s = VectorXd::Zero(m);
    VectorXd s_old(m);

    VectorXd g(n);
    VectorXd gprime(n);
    VectorXd var_g(n);
    VectorXd z(n);
    VectorXd W(n);

    for (int i_iter = 0; i_iter < max_iter; i_iter++)
    {
        g = eta.unaryExpr(std::ref(linkinv));
        var_g = g.unaryExpr(std::ref(variance));
        gprime = eta.unaryExpr(std::ref(mu_eta));
        z = eta + (y - g).cwiseQuotient(gprime);

        W = gprime.array().square() / var_g.array();
        s_old = s;
        // TODO: W.asDiagonal() performance can be improved
        // no need to create NxN matrix W.asDiagonal()
        // use something like W.colwise().sum() instead
        s = (svd.matrixU().transpose() * W.asDiagonal() * svd.matrixU()).inverse() *
            svd.matrixU().transpose() * W.asDiagonal() * z;
        eta = svd.matrixU() * s;
        if ((s - s_old).squaredNorm() < tol)
        {
            break;
        }
    }
    b = svd.matrixV() * svd.singularValues().cwiseInverse().asDiagonal() * svd.matrixU().transpose() * eta;
    VectorXd mu = eta.unaryExpr(std::ref(linkinv));
    double loglik = 0.;
    for (int i = 0; i < n; i++)
    {
        loglik += y(i) * log(mu(i)) + (1 - y(i)) * log(1 - mu(i));
    }
    return loglik;
}

/*
 * Implement repeated logistic regression
 */
VectorXd logistic_lrt(const MatrixXd &var,
                      const MatrixXd &cov,
                      const VectorXd &y,
                      int test_size,
                      const vector<int> &test_index,
                      int max_iter = 200,
                      double tol = 1e-6)
{
    int n_indiv = var.rows();
    int n_var = var.cols();
    int n_cov = cov.cols();

    MatrixXd design(n_indiv, n_cov + test_size);
    design << MatrixXd::Zero(n_indiv, test_size), cov;
    VectorXd rls_f_stat(n_var / test_size);

    // coefficients for the covariates
    VectorXd beta_cov = VectorXd::Zero(n_cov);
    logistic_reg(design(all, seq(test_size, n_cov + test_size - 1)), y, beta_cov, max_iter, tol);

    VectorXd beta_full = VectorXd::Zero(n_cov + test_size);
    VectorXd beta_reduced = VectorXd::Zero(n_cov + test_size - test_index.size());
    // find index that is in the reduced model
    vector<int> reduced_index;
    for (int i = 0; i < test_size; i++)
    {
        if (find(test_index.begin(), test_index.end(), i) == test_index.end())
        {
            reduced_index.push_back(i);
        }
    }
    for (int i = test_size; i < n_cov + test_size; i++)
    {
        reduced_index.push_back(i);
    }
    VectorXd loglik_diff(n_var / test_size);
    for (int i_test = 0; i_test < n_var / test_size; i_test++)
    {
        design(all, seq(0, test_size - 1)) =
            var(all, seq(i_test * test_size, i_test * test_size + test_size - 1));
        beta_full.setZero();
        beta_reduced.setZero();
        beta_full(seq(test_size, n_cov + test_size - 1)) = beta_cov;
        double loglik_full = logistic_reg(design, y, beta_full, max_iter, tol);
        beta_reduced = beta_full(reduced_index);
        double loglik_reduced = logistic_reg(design(all, reduced_index), y, beta_reduced, max_iter, tol);
        loglik_diff(i_test) = loglik_full - loglik_reduced;
    }
    return loglik_diff;
}

/*
 * For potential computational speedup in the linear regression mode.
b = np.dot(cov2.T, cov2)
b_inv = linalg.inv(b)

# define commonly used term in subsequent computations

# covariance terms
a = np.dot(cov1.T, cov1)
a_inv = linalg.inv(a)

v = np.dot(cov1.T, cov2)
v_mul_b_inv = np.dot(v, b_inv)
# quadratic form
b_quad = np.linalg.multi_dot([v, b_inv, v.T])
d_inv = linalg.inv(np.eye(2) - np.dot(b_quad, a_inv))

inv11 = a_inv + np.linalg.multi_dot([a_inv, b_quad, d_inv, a_inv])
inv12 = -np.dot(linalg.inv(a - quad_form), v_mul_b_inv)
inv22 = b_inv - np.linalg.multi_dot([v_mul_b_inv.T, a_inv, d_inv, v_mul_b_inv])
 */

VectorXd linear_f_test(const MatrixXd &var, const MatrixXd &cov, const VectorXd &y, int test_size,
                       const vector<int> &test_index)
{
    // F-test for linear regression
    // var: variables to be tested
    // cov: covariates included
    // y: response variables
    // test_size: number of variables included in each test.
    // test_index: index of the variables to be tested.
    int n_indiv = var.rows();
    int n_var = var.cols();
    int n_cov = cov.cols();

    MatrixXd design(n_indiv, n_cov + test_size);
    design << MatrixXd::Zero(n_indiv, test_size), cov;
    VectorXd rls_f_stat(n_var / test_size);

    for (int i_test = 0; i_test < n_var / test_size; i_test++)
    {
        design(all, seq(0, test_size - 1)) = var(all, seq(i_test * test_size, i_test * test_size + test_size - 1));
        JacobiSVD<MatrixXd> svd(design, ComputeThinU | ComputeThinV);
        VectorXd beta = svd.solve(y);
        MatrixXd ViD = svd.matrixV() * svd.singularValues().asDiagonal().inverse();
        double sigma = (y - design * beta).squaredNorm() / (n_indiv - n_cov - test_size);
        MatrixXd iXtX = ViD * ViD.transpose();
        MatrixXd f_stat = beta(test_index).transpose() * iXtX(test_index, test_index).inverse() * beta(test_index) /
                          (test_index.size() * sigma);
        rls_f_stat[i_test] = f_stat(0, 0);
    }
    return rls_f_stat;
}

PYBIND11_MODULE(admixgwas, m)
{
    m.doc() = "Genome-wide association testing in admixed population";
    m.def("linear_f_test", &linear_f_test);
    m.def("logistic_reg", &logistic_reg);
    m.def("logistic_lrt", &logistic_lrt);
}

//int main(int argc, char *argv[]) {
//    int n_indiv = 1000;
//    int n_cov = 5;
//    int n_var = 100;
//    MatrixXd cov = MatrixXd::Random(n_indiv, n_cov);
//    MatrixXd var = MatrixXd::Random(n_indiv, n_var);
//    VectorXd beta = VectorXd::Random(n_cov);
//    VectorXd mu = (cov * beta).unaryExpr(std::ref(linkinv));
//
//    VectorXd y(n_indiv);
//
//    std::random_device rd{}; // use to seed the rng
//    std::mt19937 rng{rd()}; // rng
//
//    for (int i = 0; i < n_indiv; i++){
//        std::bernoulli_distribution d(mu(i));
//        y(i) = d(rng);
//    }
//    cout << "y: " << endl;
//    cout << y({0, 1, 2, 3, 4, 5}) << endl;
//
//    VectorXd b = VectorXd::Zero(n_cov);
//    logistic_reg(cov, y, b, 100, 1e-6);
//    cout << "Groundtruth:" << endl;
//    cout << beta << endl;
//    cout << "Estimation:" << endl;
//    cout << b << endl;
//    cout << logistic_lrt(var, cov, y, 2, {0}) << endl;
//    // VectorXd y = cov * beta + VectorXd::Random(n_indiv);
//    // linear_f_test(var, cov, y, 2, {0});
//}
