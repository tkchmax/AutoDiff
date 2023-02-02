#include "Model.h"
#include "Losses.h"
#include <cassert>
#include <array>

namespace {
    std::array<std::vector<Tensor>, 2> MakeBatches(const std::vector<Tensor>& x, const std::vector<Tensor>& y, int n) {
        assert(x.size() == y.size());

        int size = ceil(double(x.size()) / n);

        const Tensor::Shape& x_shape = x[0].getShape();
        const Tensor::Shape& y_shape = y[0].getShape();

        assert(x_shape.depth == 1);
        assert(y_shape.depth == 1);

        std::vector<Tensor> X(size);
        std::vector<Tensor> Y(size);
        for (int i = 0; i < size; ++i) {
            X[i] = Tensor(x_shape.height, x_shape.width, 0);
            Y[i] = Tensor(y_shape.height, y_shape.width, 0);
            for (int k = 0; k < n; ++k) {
                X[i] = X[i].concat(x[(i * n + k) % x.size()]);
                Y[i] = Y[i].concat(y[(i * n + k) % y.size()]);
            }
        }
        return { X, Y };
    }
}

void Model::compile(ELoss loss_type, EMetrics metric_type)
{
    for (int i = 1; i < layers.size(); ++i) {
        layers[i]->build(layers[i - 1]->getOutputShape());
    }

    const Tensor::Shape& output_shape = layers.back()->getOutputShape();
    loss = std::move(std::unique_ptr<Loss>(Loss::Create(loss_type, output_shape)));
    metric = std::move(std::unique_ptr<Metric>(Metric::Create(metric_type)));
}

Tensor Model::predict(const Tensor& x)
{
    FeedDict feed_dict;
    feed_dict[*calls.begin()] = x;

    Session session;
    return session.run(calls.back(), feed_dict);
}

void Model::fit(const std::vector<Tensor>& x, const std::vector<Tensor>& y, int epochs)
{
}

void Sequential::fit(const std::vector<Tensor>& x, const std::vector<Tensor>& y, int epochs)
{
    assert(x[0].getShape() == (*layers.begin())->getOutputShape());

    //generate final computational graph
    Node* input = layers[0]->call(nullptr);
    calls = { input };
    for (int i = 1; i < layers.size(); ++i) {
        calls.push_back(layers[i]->call(calls[i - 1]));
    }

    FeedDict feed_dict;
    Session session(calls.back());
    double alpha = 0.05;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "Epoch-" << epoch + 1 << std::endl;
        for (int s = 0; s < x.size(); ++s) {

            //forward
            feed_dict[input] = x[s];
            Tensor output = session.run(feed_dict);

            //update, display metrics
            metric->update_state(output, y[s]);
            if (s > 0 && s % 1000 == 0) {
                std::cout << "#" << s << " ";
                std::cout << metric->getName() << ": ";
                std::cout << metric->result() << std::endl;
                metric->reset();
            }

            //calculate loss
            Tensor loss_value = (*loss)(output, y[s]);
            calls.back()->setGradient(loss->grad());

            //backprop
            for (int i = layers.size() - 1; i >= 1; --i) {
                Tensor db = session.grad(calls.back(), layers[i]->getBias());
                Tensor dw = session.grad(calls.back(), layers[i]->getWeights());
                layers[i]->updateWeights(dw, db, alpha);
            }
        }
    }
}
