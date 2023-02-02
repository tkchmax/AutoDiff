#include "Operation.h"
#include <stdexcept>
#include <math.h>
#include <cassert>

namespace {
    double SigmoidFunc(double x) {
        return 1.f / (1 + exp(-x));
    }
}

unsigned int Add::i = 0;
unsigned int Subtract::i = 0;
unsigned int MatMul::i = 0;
unsigned int Mul::i = 0;
unsigned int Log::i = 0;
unsigned int Pow2::i = 0;
unsigned int Minus::i = 0;
unsigned int Sigmoid::i = 0;
unsigned int Softmax::i = 0;

Tensor Add::forward(const Node::Inputs& input)
{
    if (input.size() != 2) {
        throw std::invalid_argument("Expected 2 input tensors");
    }

    const Tensor& A = input[0]->getOutput();
    const Tensor& B = input[1]->getOutput();
    return output = A + B;
}

std::vector<Tensor> Add::backward(const Tensor& grad)
{
    return { grad, grad };
}

Tensor Subtract::forward(const Node::Inputs& input)
{
    const Tensor& A = input[0]->getOutput();
    const Tensor& B = input[1]->getOutput();
    return output = A - B;
}

std::vector<Tensor> Subtract::backward(const Tensor& grad)
{
    return { grad, -grad };
}

Tensor MatMul::forward(const Node::Inputs& input)
{
    if (input.size() != 2) {
        throw std::invalid_argument("Expected 2 input tensors");
    }

    const Tensor& A = input[0]->getOutput();
    const Tensor& B = input[1]->getOutput();
    return output = A.matmul(B);
}


std::vector<Tensor> MatMul::backward(const Tensor& loss)
{
    const Tensor& A = inputs[0]->getOutput();
    const Tensor& B = inputs[1]->getOutput();
    return { loss.matmul(B.T()), A.T().matmul(loss) };
}

Tensor Mul::forward(const Node::Inputs& input)
{
    if (input.size() != 2) {
        throw std::invalid_argument("Expected 2 input tensors");
    }

    const Tensor& A = inputs[0]->getOutput();
    const Tensor& B = inputs[1]->getOutput();
    return output = A * B;
}

std::vector<Tensor> Mul::backward(const Tensor& loss)
{
    const Tensor& A = inputs[0]->getOutput();
    const Tensor& B = inputs[1]->getOutput();
    return { B * loss, A * loss };
}

Tensor Log::forward(const Node::Inputs& inputs)
{
    if (inputs.size() != 1) {
        throw std::invalid_argument("Expected only one input tensor");
    }

    const Tensor& input = inputs[0]->getOutput();
    output = Tensor(input.getShape());
    for (int i = 0; i < input.size(); ++i) {
        output[i] = log(input[i]);
    }

    return output;
}

Tensor Pow2::forward(const Node::Inputs& inputs)
{
    if (inputs.size() != 1) {
        throw std::invalid_argument("Expected only one input tensor");
    }
    const Tensor& A = inputs[0]->getOutput();

    return output = A.pow2();

}

std::vector<Tensor> Pow2::backward(const Tensor& loss)
{
    const Tensor& A = inputs[0]->getOutput();
    return { (A * 2) * loss };
}

std::vector<Tensor> Log::backward(const Tensor& loss)
{
    const Tensor& A = inputs[0]->getOutput();
    return { (1 / A) * loss };
}

Tensor Minus::forward(const Node::Inputs& inputs)
{
    if (inputs.size() != 1) {
        throw std::invalid_argument("Expected only one input tensor");
    }

    const Tensor& A = inputs[0]->getOutput();
    return output = -A;
}


std::vector<Tensor> Minus::backward(const Tensor& loss)
{
    return { -loss };
}

Tensor Sigmoid::forward(const Node::Inputs& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("Expected only one input tensor");
    }

    const Tensor& A = inputs[0]->getOutput();
    Tensor sigm(A.getShape());
    for (int i = 0; i < sigm.size(); ++i) {
        sigm[i] = SigmoidFunc(A[i]);
    }

    return output = sigm;
}

std::vector<Tensor> Sigmoid::backward(const Tensor& loss) {
    assert(loss.getShape() == output.getShape());

    Tensor grad(output.getShape());
    for (int i = 0; i < output.size(); ++i) {
        grad[i] = SigmoidFunc(output[i]) * (1 - SigmoidFunc(output[i]));
    }

    return { grad * loss };
}

Tensor Softmax::forward(const Node::Inputs& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("Expected only one input tensor");
    }

    const Tensor& A = inputs[0]->getOutput();
    const Tensor::Shape& shape = A.getShape();

    double sum = 0;
    Tensor res(shape);

    assert(shape.width == 1);
    for (int d = 0; d < shape.depth; ++d) {
        double sum = 0;
        for (int i = 0; i < shape.height; ++i) {
            res(i, 0, d) = exp(A[i]);
            sum += res(i, 0, d);
        }
        for (int i = 0; i < shape.height; ++i) {
            res(i, 0, d) /= sum;
        }
    }

    return output = res;
}

std::vector<Tensor> Softmax::backward(const Tensor& loss) {
    assert(loss.getShape() == output.getShape());

    const Tensor::Shape& out_shape = output.getShape();
    Tensor dx(out_shape);
    for (int d = 0; d < out_shape.depth; ++d) {
        for (int i = 0; i < out_shape.height; ++i) {
            for (int j = 0; j < out_shape.height; ++j) {
                double temp = (i == j) ?
                    output(i, 0, d) * (1 - output(i, 0, d)) :
                    -output(j, 0, d) * output(i, 0, d);
                temp *= loss(j, 0, d);
                dx(i, 0, d) += temp;
            }
        }
    }
    return { dx };
}