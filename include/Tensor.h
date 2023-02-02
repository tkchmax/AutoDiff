#pragma once
#ifndef TENSOR_H_
#define TENSOR_H_

#include <vector>
#include <string>
#include <iostream>

typedef std::vector<double> Mat;
typedef std::vector<std::vector<double>> Mat2;
typedef std::vector<std::vector<std::vector<double>>> Mat3;

class Tensor
{
public:
    struct Shape {
        Shape() : Shape(0, 0, 0) {}
        Shape(int h, int w, int d = 1) : height(h), width(w), depth(d) {}
        bool operator==(const Shape& rsh) const;
        bool operator!=(const Shape& rsh) const;
        int operator[](int idx) const;
        int& operator[](int idx);
        int height;
        int width;
        int depth;
    };
public:
    Tensor() : shape(Tensor::Shape({ 0, 0, 0 })) {}
    Tensor(const Tensor::Shape& shape);
    Tensor(int h, int w, int d = 1) : Tensor(Tensor::Shape(h, w, d)) {}
    Tensor(const Tensor::Shape& shape, double val);
    Tensor(const Tensor::Shape& shape, double mean, double stddiv);
    Tensor(const Mat3& mat);
    Tensor(const Mat2& mat);
    Tensor(const Mat& mat);
    Tensor(double scalar);
    const Tensor::Shape& getShape() const;
    int size() const;
    void clear();
    const Mat& getRawValues() const;
    Tensor matmul(const Tensor& rsh) const;
    Tensor T() const;
    Tensor diag() const;
    Tensor outer(const Tensor& rsh) const;
    Tensor sum() const;
    double Sum() const;
    Tensor pow2() const;
    Tensor concat(const Tensor& other) const;
    Mat3 getValues();
    double operator()(int i, int j, int d) const;
    double& operator()(int i, int j, int d);
    double operator[](int idx) const;
    double& operator[](int idx);
    void operator+=(const Tensor& rsh);
    Tensor operator+(const Tensor& rsh) const;
    Tensor operator-(const Tensor& rsh) const;
    Tensor operator-(double rsh) const;
    Tensor operator-() const;
    void operator-=(const Tensor& rsh);
    Tensor operator*(const Tensor& rsh) const;
    Tensor operator*(double rsh) const;
    Tensor operator/(const Tensor& rsh) const;
    friend std::ostream& operator<<(std::ostream& out, const Tensor& tensor);
    friend Tensor operator-(double val, const Tensor& rsh);
    friend Tensor operator/(double val, const Tensor& rsh);
    static Tensor ones(const Tensor::Shape& shape);
private:
    Tensor::Shape shape;
    Mat values;
};

#endif
