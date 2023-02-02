#pragma once
#ifndef LOSSES_H_
#define LOSSES_H_

#include "Session.h"
#include "NN.h"

enum class ELoss {
    CROSS_ENTROPY,
    MSE
};

class Loss {
public:
    //static Loss Create(ELoss loss, const Tensor::Shape& output_shape);
    static Loss* Create(ELoss loss, const Tensor::Shape& output_shape);
    virtual Tensor operator()(const Tensor& prediction, const Tensor& ground_truth);
    Tensor grad();
protected:
    NN nn;
    Session session;
};

class CrossEntropyLoss : public Loss {
public:
    CrossEntropyLoss(const Tensor::Shape& output_shape);
};

class MSELoss : public Loss {
public:
    MSELoss(const Tensor::Shape& output_shape);
};

#endif