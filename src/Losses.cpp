#include "Losses.h"

CrossEntropyLoss::CrossEntropyLoss(const Tensor::Shape& output_shape)
{
    auto ground_truth = nn.placeholder(output_shape, "ground_truth");
    auto predicted = nn.placeholder(output_shape, "predicted");
    auto ONE = nn.constant(Tensor(output_shape, 1));
    auto loss =
        nn.subtract(
            nn.mul(nn.minus(ground_truth), nn.log(predicted)),
            nn.mul(nn.subtract(ONE, ground_truth), nn.log(nn.subtract(ONE, predicted)))
        );

    loss->setName("loss");
    loss->setGradient(Tensor(output_shape, 1));
}

Loss* Loss::Create(ELoss loss, const Tensor::Shape& output_shape)
{
    switch (loss) {
    case ELoss::CROSS_ENTROPY:
        return new CrossEntropyLoss(output_shape);
    case ELoss::MSE:
        return new MSELoss(output_shape);
    default:
        throw;
    }
}

Tensor Loss::operator()(const Tensor& pred, const Tensor& truth)
{
    FeedDict feed_dict;

    auto predicted = nn.findByName("predicted");
    auto ground_truth = nn.findByName("ground_truth");
    auto loss = nn.findByName("loss");

    feed_dict[predicted] = pred;
    feed_dict[ground_truth] = truth;

    return session.run(loss, feed_dict);
}

Tensor Loss::grad()
{
    auto loss = nn.findByName("loss");
    auto predicted = nn.findByName("predicted");

    return session.grad(loss, predicted);
}

MSELoss::MSELoss(const Tensor::Shape& output_shape)
{
    auto ground_truth = nn.placeholder(output_shape, "ground_truth");
    auto predicted = nn.placeholder(output_shape, "predicted");

    auto loss = nn.pow2(nn.subtract(ground_truth, predicted));

    loss->setName("loss");
    loss->setGradient(Tensor(output_shape, 1));
}
