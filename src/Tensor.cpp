#include "Tensor.h"
#include <stdexcept>
#include <random>

namespace {
    double NRand(double mean, double stddev)
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::normal_distribution<double> nd(mean, stddev);
        return nd(gen);
    }
}

Tensor::Tensor(const Tensor::Shape& shape, double val)
{
    Tensor temp(shape);
    for (auto& el : temp.values) {
        el = val;
    }
    *this = temp;
}

Tensor::Tensor(const Tensor::Shape& shape, double mean, double stddiv)
{
    Tensor tmp(shape);
    for (int i = 0; i < tmp.size(); ++i) {
        tmp[i] = NRand(mean, stddiv);
    }
    *this = tmp;
}

Tensor::Tensor(const Mat3& mat)
{
    shape.depth = mat.size();
    shape.height = mat[0].size();
    shape.width = mat[0][0].size();
    values.resize(shape.depth * shape.height * shape.width);
    for (int d = 0; d < shape.depth; ++d) {
        for (int h = 0; h < shape.height; ++h) {
            for (int w = 0; w < shape.width; ++w) {
                this->operator()(h, w, d) = mat[d][h][w];
            }
        }
    }
}

Tensor::Tensor(const Mat2& mat)
{
    Tensor temp(Tensor::Shape(mat.size(), mat[0].size(), 1));
    for (int i = 0; i < mat.size(); ++i) {
        for (int j = 0; j < mat[0].size(); ++j) {
            temp(i, j, 0) = mat[i][j];
        }
    }
    *this = temp;
}

Tensor::Tensor(const Mat& mat)
{
    Tensor temp(Tensor::Shape(1, mat.size(), 1));
    for (int i = 0; i < mat.size(); ++i) {
        temp(0, i, 0) = mat[i];
    }
    *this = temp;
}

Tensor::Tensor(double scalar)
{
    shape = Tensor::Shape(1, 1, 1);
    values.resize(1);
    values[0] = scalar;
}

Tensor::Tensor(const Tensor::Shape& shape)
{
    this->shape = shape;
    values.resize(shape.height * shape.width * shape.depth);

}

const Tensor::Shape& Tensor::getShape() const
{
    return shape;
}

int Tensor::size() const
{
    return values.size();
}

void Tensor::clear()
{
    values.clear();
}

const Mat& Tensor::getRawValues() const
{
    return values;
}

Mat3 Tensor::getValues()
{
    return Mat3();
}

double& Tensor::operator()(int i, int j, int d)
{
    if (shape.width == 1 && shape.height > 1) {
        return values[d * shape.height + i];
    }
    return values[d * shape.width * shape.height + i * shape.width + j];
}

void Tensor::operator+=(const Tensor& rsh)
{
    if (shape != rsh.shape) {
        throw std::invalid_argument("shapes not equal");
    }

    for (int i = 0; i < values.size(); ++i) {
        values[i] += rsh.values[i];
    }
}

double& Tensor::operator[](int idx)
{
    return values[idx];
}

double Tensor::operator()(int i, int j, int d) const
{
    if (shape.width == 1 && shape.height > 1) {
        return values[d * shape.height + i];
    }
    return values[d * shape.width * shape.height + i * shape.width + j];
}

double Tensor::operator[](int idx) const
{
    return values[idx];
}

Tensor Tensor::operator+(const Tensor& rsh) const
{
    if (shape != rsh.shape) {
        throw std::invalid_argument("shapes are not equal");
    }

    Tensor sum(shape);
    for (int i = 0; i < values.size(); ++i) {
        sum[i] = values[i] + rsh[i];
    }
    return sum;
}

Tensor Tensor::operator-(const Tensor& rsh) const
{
    if (shape != rsh.shape) {
        throw std::invalid_argument("shapes are not equal");
    }

    Tensor diff(shape);
    for (int i = 0; i < values.size(); ++i) {
        diff[i] = values[i] - rsh[i];
    }
    return diff;
}

Tensor Tensor::operator-(double rsh) const
{
    Tensor diff(shape);
    for (int i = 0; i < values.size(); ++i) {
        diff[i] = values[i] - rsh;
    }
    return diff;
}

Tensor Tensor::operator-() const
{
    Tensor minus(shape);
    for (int i = 0; i < values.size(); ++i) {
        minus[i] = -values[i];
    }
    return minus;
}

void Tensor::operator-=(const Tensor& rsh)
{
    *this = *this - rsh;
}

Tensor Tensor::operator*(const Tensor& rsh) const
{
    if (shape != rsh.shape) {
        throw std::invalid_argument("shapes are not equal");
    }

    Tensor res(shape);
    for (int i = 0; i < values.size(); ++i) {
        res[i] = values[i] * rsh.values[i];
    }
    return res;
}

Tensor Tensor::operator*(double rsh) const
{
    Tensor res(shape);
    for (int i = 0; i < values.size(); ++i) {
        res[i] = values[i] * rsh;
    }
    return res;
}

Tensor Tensor::operator/(const Tensor& rsh) const
{
    if (shape != rsh.shape) {
        throw std::invalid_argument("shapes are not equal");
    }

    Tensor div(shape);
    for (int i = 0; i < values.size(); ++i) {
        div[i] = values[i] / rsh[i];
    }
    return div;
}

Tensor Tensor::matmul(const Tensor& rsh) const
{
    Tensor::Shape rsh_shape = rsh.getShape();
    if (shape.depth != rsh_shape.depth) {
        throw std::invalid_argument("Tensor1 dim not equal Tensor2 dim");
    }
    if (shape.width != rsh_shape.height) {
        throw std::invalid_argument("Tensor1 column not equal Tensor2 rows");
    }

    Tensor matmul(Tensor::Shape(shape.height, rsh_shape.width, rsh_shape.depth));
    for (int d = 0; d < shape.depth; ++d) {
        for (int i = 0; i < shape.height; ++i) {
            for (int j = 0; j < rsh_shape.width; ++j) {
                for (int k = 0; k < shape.width; ++k) {
                    matmul(i, j, d) += this->operator()(i, k, d) * rsh(k, j, d);
                }
            }
        }
    }

    return matmul;
}

Tensor Tensor::T() const
{
    Mat3 temp = Mat3(shape.depth, Mat2(shape.width, Mat(shape.height)));
    for (int d = 0; d < shape.depth; ++d) {
        for (int i = 0; i < shape.height; ++i) {
            for (int j = 0; j < shape.width; ++j) {
                temp[d][j][i] = operator()(i, j, d);
            }
        }
    }
    return Tensor(temp);
}

Tensor Tensor::diag() const
{
    Tensor::Shape diagShape(shape);
    if (shape.width < shape.height) {
        diagShape.height = shape.width;
    }
    Tensor diag(diagShape);
    for (int d = 0; d < shape.depth; ++d) {
        for (int i = 0; i < shape.height && i < shape.width; ++i) {
            diag(i, i, d) = this->operator()(i, i, d);
        }
    }
    return diag;
}

Tensor Tensor::outer(const Tensor& rsh) const
{
    return this->matmul(rsh.T());
}

Tensor Tensor::sum() const
{
    double sum = 0;
    for (int i = 0; i < values.size(); ++i) {
        sum += values[i];
    }
    return sum;
}

double Tensor::Sum() const
{
    return sum().values[0];
}

Tensor Tensor::pow2() const
{
    Tensor pow(shape);
    for (int i = 0; i < values.size(); ++i) {
        pow[i] = values[i] * values[i];
    }
    return pow;
}

Tensor Tensor::concat(const Tensor& other) const
{
    const Tensor::Shape& other_shape = other.getShape();
    if (shape.width != other_shape.width || shape.height != other_shape.height) {
        throw std::invalid_argument("shapes not equal");
    }

    Tensor concatenated(shape.height, shape.width, shape.depth + other_shape.depth);

    std::copy(values.begin(), values.end(), concatenated.values.begin());
    std::copy(other.values.begin(), other.values.end(), concatenated.values.begin() + values.size());

    return concatenated;

}

Tensor Tensor::ones(const Tensor::Shape& shape)
{
    Tensor tensor(shape);
    for (int i = 0; i < tensor.size(); ++i) {
        tensor[i] = 1;
    }
    return tensor;
}

bool Tensor::Shape::operator==(const Shape& rsh) const
{
    return
        height == rsh.height &&
        width == rsh.width &&
        depth == rsh.depth;
}

bool Tensor::Shape::operator!=(const Shape& rsh) const
{
    return !((*this) == rsh);
}

int Tensor::Shape::operator[](int idx) const
{
    switch (idx) {
    case 0: return height;
    case 1: return width;
    case 2: return depth;
    default: throw;
    }
}

int& Tensor::Shape::operator[](int idx)
{
    switch (idx) {
    case 0: return height;
    case 1: return width;
    case 2: return depth;
    default: throw;
    }
}

std::ostream& operator<<(std::ostream& out, const Tensor& tensor)
{
    for (int d = 0; d < tensor.shape.depth; ++d) {
        for (int i = 0; i < tensor.shape.height; ++i) {
            for (int j = 0; j < tensor.shape.width; ++j) {
                out << tensor(i, j, d) << " ";
            }
            out << std::endl;
        }
        out << std::endl;
    }

    return out;
}

Tensor operator-(double val, const Tensor& rsh)
{
    Tensor substr(rsh.getShape());
    for (int i = 0; i < rsh.size(); ++i) {
        substr[i] = val - rsh[i];
    }
    return substr;
}

Tensor operator/(double val, const Tensor& rsh)
{
    Tensor div(rsh.getShape());
    for (int i = 0; i < rsh.size(); ++i) {
        div[i] = val / rsh[i];
    }
    return div;
}
