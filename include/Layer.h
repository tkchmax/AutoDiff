#pragma once
#ifndef LAYER_H_
#define LAYER_H_

#include "NN.h"

enum class EActivation {
    relu,
    sigmoid,
    softmax
};

class Layer
{
public:
    Layer(int units) : units(units) {}
    virtual void build(const Tensor::Shape& input_shape) = 0;
    virtual Node* call(Node* input) = 0;
    const Tensor::Shape& getOutputShape() const;
    Variable* getWeights();
    Variable* getBias();
    void updateWeights(const Tensor& dw, const Tensor& db, double alpha);

protected:
    Variable weights;
    Variable bias;
    Tensor::Shape output_shape;
    NN nn;
    int units;
};

class Input : public Layer {
public:
    Input(const Tensor::Shape& input_shape);
    void build(const Tensor::Shape& input_shape) override;
    Node* call(Node* input=nullptr) override;
    void resize(const Tensor::Shape& new_shape);
private:
    Placeholder value;
};

class DenseLayer : public Layer {
public:
    DenseLayer(int units, EActivation activation);
    void build(const Tensor::Shape& input_shape) override;
    Node* call(Node* input) override;
private:
    EActivation activation;
};

#endif
