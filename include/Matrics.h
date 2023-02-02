#pragma once
#ifndef METRICS_H_
#define METRICS_H_

#include "Tensor.h"

enum class EMetrics {
    MSE,
    ACCURACY
};

class Metric
{
public:
    Metric(EMetrics type) : value(0), type(type) {}
    static Metric* Create(EMetrics metric);
    virtual void update_state(const Tensor& output, const Tensor& label) = 0;
    virtual void reset();
    double result();
    std::string getName();
protected:
    double value;
    EMetrics type;
};

class MSEMetric : public Metric {
public:
    MSEMetric() : Metric(EMetrics::MSE) {}
    void update_state(const Tensor& output, const Tensor& label) override;
};

class AccuracyMetric : public Metric {
public:
    AccuracyMetric() : Metric(EMetrics::ACCURACY), count(0), corrects(0) {}
    void update_state(const Tensor& output, const Tensor& label) override;
    void reset() override;
private:
    unsigned int count;
    unsigned int corrects;
};

#endif
