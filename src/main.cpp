#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <ctime>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace std;
using namespace Eigen;
using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/*
 * Logistic regression IRLS implementation.
 * Based on https://bwlewis.github.io/GLM/#svdnewton
 * Perform likelihood ratio test for hypothesis testing
 */

/*
 * The following functions follow the interface for GLM, currently only logistic
 * regression is implemented. However, it is straightforward to extend to other
 * GLM family.
 */

double link(double mu) { return log(mu / (1 - mu)); }

double linkinv(double eta) { return 1 / (1 + exp(-eta)); }

double variance(double g) { return g * (1 - g); }

double mu_eta(double eta) {
  // derivative of mu to eta
  return exp(-eta) / pow((1 + exp(-eta)), 2);
}

double logistic_reg(const MatrixXd &X, const VectorXd &y,
                    Eigen::Ref<VectorXd> b, Eigen::Ref<VectorXd> se,
                    int max_iter, double tol) {
  // b is the effect sizes to be optimized in place, also used as the starting
  // points return loglikelihood

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

  for (int i_iter = 0; i_iter < max_iter; i_iter++) {
    g = eta.unaryExpr(std::ref(linkinv));
    var_g = g.unaryExpr(std::ref(variance));
    gprime = eta.unaryExpr(std::ref(mu_eta));
    z = eta + (y - g).cwiseQuotient(gprime);
    W = gprime.array().square() / var_g.array();

    // fraction of observations can be perfectly predicted
    double predclose_frac =
        ((y - g).cwiseAbs().array() < 1e-4).cast<double>().mean();
    if (predclose_frac > 0.05) {
      cout << "Warning: [tinygwas.logistic_reg] A fraction of "
           << static_cast<int>(predclose_frac * 100)
           << "% of the observations can be perfectly predicted (abs(y - g) < "
              "1e-4). This might indicate complete quasi-separation. Exiting "
              "iterations early and the corresponding loglik is set to NaN"
           << endl;
      return std::numeric_limits<double>::quiet_NaN();
    }

    s_old = s;

    // solve U.T W U s = U.T W z
    auto UtW = svd.matrixU().transpose() * W.asDiagonal();
    s = (UtW * svd.matrixU()).ldlt().solve(UtW * z);
    eta = svd.matrixU() * s;
    if ((s - s_old).squaredNorm() < tol) {
      break;
    }
  }

  b = svd.matrixV() * svd.singularValues().cwiseInverse().asDiagonal() *
      svd.matrixU().transpose() * eta;
  VectorXd mu = eta.unaryExpr(std::ref(linkinv));

  // calculate standard errors of beta
  VectorXd var_mu = mu.unaryExpr(std::ref(variance));
  se = (X.transpose() * var_mu.asDiagonal() * X)
           .inverse()
           .diagonal()
           .cwiseSqrt();

  double loglik = 0.;
  for (int i = 0; i < n; i++) {
    loglik += y(i) * log(mu(i) + 1e-20) + (1 - y(i)) * log(1 - mu(i) + 1e-20);
  }
  return loglik;
}

/*
 * Implement repeated logistic regression
 */
void logistic_lrt(const MatrixXd &var, const MatrixXd &cov, const VectorXd &y,
                  int test_size, const vector<int> &test_index,
                  Eigen::Ref<RowMatrixXd> res, int max_iter = 200,
                  double tol = 1e-6) {
  // return BETA1, SE1, BETA2, SE2, N, P
  int n_indiv = var.rows();
  int n_var = var.cols();
  int n_cov = cov.cols();

  int n_test = n_var / test_size;

  if (!((res.rows() == n_test) && (res.cols() == test_size * 2 + 2))) {
    throw std::invalid_argument("[tinygwas.logistic_lrt] res should be a "
                                "matrix of size (n_test, test_size * 2 + 2)");
  }

  MatrixXd design(n_indiv, n_cov + test_size);
  design << MatrixXd::Zero(n_indiv, test_size), cov;

  // coefficients for the covariates
  VectorXd beta_cov = VectorXd::Zero(n_cov);
  VectorXd se_cov = VectorXd::Zero(n_cov);
  logistic_reg(design(all, seq(test_size, n_cov + test_size - 1)), y, beta_cov,
               se_cov, max_iter, tol);

  VectorXd beta_full = VectorXd::Zero(n_cov + test_size);
  VectorXd se_full = VectorXd::Zero(n_cov + test_size);
  VectorXd beta_reduced = VectorXd::Zero(n_cov + test_size - test_index.size());
  VectorXd se_reduced = VectorXd::Zero(n_cov + test_size - test_index.size());

  // find index that is in the reduced model
  vector<int> reduced_index;
  for (int i = 0; i < test_size; i++) {
    if (find(test_index.begin(), test_index.end(), i) == test_index.end()) {
      reduced_index.push_back(i);
    }
  }
  for (int i = test_size; i < n_cov + test_size; i++) {
    reduced_index.push_back(i);
  }
  VectorXd loglik_diff(n_test);

  for (int i_test = 0; i_test < n_test; i_test++) {
    design(all, seq(0, test_size - 1)) =
        var(all, seq(i_test * test_size, i_test * test_size + test_size - 1));

    // get the individual where all the variables are not NaN
    vector<int> test_indiv_idx;
    for (int i = 0; i < n_indiv; i++) {
      if (!design(i, seq(0, test_size - 1)).array().isNaN().any()) {
        test_indiv_idx.push_back(i);
      }
    }
    int n_test_indiv = test_indiv_idx.size();

    beta_full.setZero();
    beta_reduced.setZero();
    beta_full(seq(test_size, n_cov + test_size - 1)) = beta_cov;

    double loglik_full =
        logistic_reg(design(test_indiv_idx, all), y(test_indiv_idx), beta_full,
                     se_full, max_iter, tol);
    beta_reduced = beta_full(reduced_index);
    double loglik_reduced =
        logistic_reg(design(test_indiv_idx, reduced_index), y(test_indiv_idx),
                     beta_reduced, se_reduced, max_iter, tol);

    // save results: BETA1, SE1, BETA2, SE2, ..., N, loglihood_diff
    for (int i = 0; i < test_size; i++) {
      res(i_test, i * 2) = beta_full(i);
      res(i_test, i * 2 + 1) = se_full(i);
    }
    res(i_test, test_size * 2) = n_test_indiv;
    res(i_test, test_size * 2 + 1) = loglik_full - loglik_reduced;
  }
  return;
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

void linear_f_test(const MatrixXd &var, const MatrixXd &cov, const VectorXd &y,
                   int test_size, const vector<int> &test_index,
                   Eigen::Ref<RowMatrixXd> res) {
  // F-test for linear regression
  // var: variables to be tested (NaN is allowed)
  // cov: covariates common to all tests (no NaN is assumed)
  // y: response variables
  // test_size: number of variables included in each test.
  // test_index: index of the variables to be tested.
  // return: BETA1, SE1, BETA2, SE2, N, f-stat

  int n_indiv = var.rows();
  int n_var = var.cols();
  int n_cov = cov.cols();

  MatrixXd design(n_indiv, n_cov + test_size);
  design << MatrixXd::Zero(n_indiv, test_size), cov;
  int n_test = n_var / test_size;
  if (!((res.rows() == n_test) && (res.cols() == test_size * 2 + 2))) {
    throw std::invalid_argument("[tinygwas.logistic_lrt] res should be a "
                                "matrix of size (n_test, test_size * 2 + 2)");
  }

  for (int i_test = 0; i_test < n_test; i_test++) {
    design(all, seq(0, test_size - 1)) =
        var(all, seq(i_test * test_size, i_test * test_size + test_size - 1));

    // get the individual where all the variables are not NaN
    vector<int> test_indiv_idx;
    for (int i = 0; i < n_indiv; i++) {
      if (!design(i, seq(0, test_size - 1)).array().isNaN().any()) {
        test_indiv_idx.push_back(i);
      }
    }
    int n_test_indiv = test_indiv_idx.size();

    // compute only on the individuals that do not have NaN
    JacobiSVD<MatrixXd> svd(design(test_indiv_idx, all),
                            ComputeThinU | ComputeThinV);
    VectorXd beta = svd.solve(y(test_indiv_idx));
    MatrixXd ViD = svd.matrixV() * svd.singularValues().asDiagonal().inverse();
    double sigmasq =
        (y(test_indiv_idx) - design(test_indiv_idx, all) * beta).squaredNorm() /
        (n_test_indiv - n_cov - test_size);
    MatrixXd iXtX = ViD * ViD.transpose();
    MatrixXd f_stat = beta(test_index).transpose() *
                      iXtX(test_index, test_index).inverse() *
                      beta(test_index) / (test_index.size() * sigmasq);

    // save results: BETA1, SE1, BETA2, SE2, ..., N, f-stat
    for (int i = 0; i < test_size; i++) {
      res(i_test, i * 2) = beta(i);
      res(i_test, i * 2 + 1) = sqrt(sigmasq * iXtX(i, i));
    }
    res(i_test, test_size * 2) = n_test_indiv;
    res(i_test, test_size * 2 + 1) = f_stat(0, 0);
  }
  return;
}

PYBIND11_MODULE(tinygwas, m) {
  m.doc() = "Toy C++ implementation with python wrapper for genome-wide "
            "association testing";
  m.def("linear_f_test", &linear_f_test);
  m.def("logistic_reg", &logistic_reg);
  m.def("logistic_lrt", &logistic_lrt);
}
