#ifndef FILEUTILS_H
#define FILEUTILS_H
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

MatrixXi read_int_mat(const std::string &path);
VectorXi read_pos(const std::string &path);
void write_int_mat(const std::string &path, const MatrixXi& mat);
bool is_file_exist(const string & fileName);

#endif
