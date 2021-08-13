#include "FileUtils.h"

bool is_file_exist(const string & fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

MatrixXi read_int_mat(const std::string &path) {
    std::ifstream in_data;
    in_data.open(path);
    std::string line;
    std::vector<int> values;
    uint n_row = 0;

    while (std::getline(in_data, line)) {
        for (int i = 0; i < line.size(); i++) {
            values.push_back(line[i] - '0');
        }
        n_row++;
    }
    return Eigen::Map<Matrix<int, Dynamic, Dynamic, RowMajor>>(values.data(), n_row, values.size() / n_row);
}

void write_int_mat(const std::string &path, const MatrixXi& mat) {
    ofstream output(path);
    for (int i = 0; i < mat.rows(); i++){
        for (int j = 0; j < mat.cols(); j++){
            output << mat(i, j);
        }
        output << "\n";
    }
}

VectorXi read_pos(const std::string &path) {
    std::ifstream in_data;
    in_data.open(path);
    std::string line;
    std::vector<int> values;

    while (std::getline(in_data, line)) {
        values.push_back(std::stoi(line));
    }
    return Map<VectorXi>(values.data(), values.size());
}
