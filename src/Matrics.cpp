#include "Matrics.h"
#include <cassert>

namespace {
    unsigned int argmax(const Tensor& tensor) {
        double max = tensor(0, 0, 0);
        unsigned int idx = 0;
        for (int i = 0; i < tensor.getShape().height; ++i) {
            if (tensor(i, 0, 0) > max) {
                max = tensor(i, 0, 0);
                idx = i;
            }
        }
        return idx;
    }
}

Metric* Metric::Create(EMetrics metric)
{
    switch (metric) {
    case EMetrics::MSE:
        return new MSEMetric();
    case EMetrics::ACCURACY:
        return new AccuracyMetric();
    default:
        throw;
    }
}

double Metric::result()
{
    return value;
}

std::string Metric::getName()
{
    switch (type) {
    case EMetrics::ACCURACY: return "Accuracy";
    case EMetrics::MSE: return "MSE";
    default: throw;
    }
}

void Metric::reset()
{
    value = 0;
}

void MSEMetric::update_state(const Tensor& output, const Tensor& label)
{
    value = (label - output).pow2().Sum();
}

void AccuracyMetric::update_state(const Tensor& output, const Tensor& label)
{
    assert(output.getShape() == label.getShape());
    count += 1;
    corrects += (argmax(output) == argmax(label)) ? 1 : 0;
    value = double(corrects) / count;
}

void AccuracyMetric::reset()
{
    value = count = corrects = 0;
}
