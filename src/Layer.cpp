#include "Layer.h"

DenseLayer::DenseLayer(int units, EActivation activation) : Layer(units)
{
    this->activation = activation;
}

void DenseLayer::build(const Tensor::Shape& input_shape)
{
    Tensor w(Tensor::Shape(units, input_shape[0], 1), 0, 1);
    Tensor b(Tensor::Shape(units, 1, 1), 0, 1);

    weights = Variable(w);
    bias = Variable(b);

    output_shape = Tensor::Shape(units, 1, 1);
}

Node* DenseLayer::call(Node* input)
{
    nn.reset();
    auto net = nn.add(nn.matmul(&weights, input), &bias);

    switch (activation) {
    case EActivation::sigmoid: return nn.sigmoid(net);
    case EActivation::softmax: return nn.softmax(net);
    default:
        throw;
    }
}

Input::Input(const Tensor::Shape& input_shape) : Layer(1), value(input_shape)
{
    output_shape = input_shape;
}

void Input::build(const Tensor::Shape& input_shape)
{
}

void Input::resize(const Tensor::Shape& new_shape)
{
    output_shape = new_shape;
    value.resize(new_shape);
}

Node* Input::call(Node* input)
{
    return &value;
}

const Tensor::Shape& Layer::getOutputShape() const
{
    return output_shape;
}

Variable* Layer::getWeights()
{
    return &weights;
}

Variable* Layer::getBias()
{
    return &bias;
}

void Layer::updateWeights(const Tensor& dw, const Tensor& db, double alpha)
{
    weights.assign(weights.getOutput() - dw * alpha);
    bias.assign(bias.getOutput() - db * alpha);
}
