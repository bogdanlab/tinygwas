import statsmodels.api as sm
import numpy as np
import tinygwas

# statsmodels implementation
def linear_ftest(var, cov, pheno, var_size, test_vars):
    """
    Same interface with the tinygwas.linear_f_test function.
    """
    n_indiv = var.shape[0]
    n_var = var.shape[1]
    n_cov = cov.shape[1]

    design = np.zeros((n_indiv, var_size + n_cov))
    design[:, var_size : var_size + n_cov] = cov

    n_test = int(n_var / var_size)
    fvalues = np.zeros(n_test)

    f_test_r_matrix = np.zeros((len(test_vars), design.shape[1]))
    for i, v in enumerate(test_vars):
        f_test_r_matrix[i, v] = 1

    for i_test in range(n_test):
        design[:, 0:var_size] = var[:, i_test * var_size : (i_test + 1) * var_size]
        model = sm.OLS(pheno, design, missing="drop").fit()
        fvalues[i_test] = model.f_test(f_test_r_matrix).fvalue
    return fvalues


def test_consistency():
    np.random.seed(1234)
    n_indiv = 100
    n_cov = 3
    var_size = 3
    n_snp = 10
    test_vars = [1, 2]
    cov = np.random.normal(size=(n_indiv, n_cov))
    pheno = np.random.normal(size=n_indiv)
    var = np.random.normal(size=(n_indiv, n_snp * var_size))
    # test consistency without NaN
    fvalues1 = np.empty(n_snp)
    nindiv1 = np.empty(n_snp)
    tinygwas.linear_f_test(var, cov, pheno, var_size, test_vars, fvalues1, nindiv1)
    fvalues2 = linear_ftest(var, cov, pheno, var_size, test_vars)
    print("Without NaN:")
    print(fvalues1[0:5])

    print("With NaN:")

    # test consistency with NaN
    n_rand = 10
    # randomly set two values to NaN
    for i in range(n_rand):
        nan_var = var.copy()
        for _ in range(3):
            nan_var[
                np.random.randint(0, nan_var.shape[0]),
                np.random.randint(0, nan_var.shape[1]),
            ] = np.nan
        fvalues1 = np.empty(n_snp)
        nindiv1 = np.empty(n_snp)
        tinygwas.linear_f_test(
            nan_var, cov, pheno, var_size, test_vars, fvalues1, nindiv1
        )
        fvalues2 = linear_ftest(nan_var, cov, pheno, var_size, test_vars)
        print(fvalues1[0:5])

        assert np.allclose(fvalues1, fvalues2)