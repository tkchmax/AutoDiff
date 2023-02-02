#pragma once
#ifndef MODEL_H_
#define MODEL_H_

#include "Layer.h"
#include "Losses.h"
#include "Matrics.h"

class Model
{
public:
    void compile(ELoss loss, EMetrics metric);
    Tensor predict(const Tensor& x);
    virtual void fit(const std::vector<Tensor>& x, const std::vector<Tensor>& y, int epochs);

protected:
    std::vector<std::unique_ptr<Layer>> layers;
    std::vector<Node*> calls;
    std::unique_ptr<Loss> loss;
    std::unique_ptr<Metric> metric;
};

class Sequential : public Model {
public:
    Sequential() {}
    template<typename... Args>
    Sequential(Args... args) { 
        initLayers(args...);
    }
    void fit(const std::vector<Tensor>& x, const std::vector<Tensor>& y, int epochs) override;
private:
    template<typename T, typename... Args>
    void initLayers (T l, Args... args) {
        layers.push_back(std::unique_ptr<Layer>{new T(l)});
        initLayers(args...);
    }
    void initLayers(){}
};

#endif
