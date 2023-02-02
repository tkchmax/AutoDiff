#include "NN.h"
#include <algorithm>

NN::NN()
{
    graph = Graph();
}

NN& NN::ByDefault()
{
    static NN instance;
    return instance;
}

void NN::reset()
{
    graph.reset();
}

Node* NN::findByName(const std::string& name) const
{
    const auto& nodes = graph.getNodes();
    auto it = std::find_if(nodes.begin(), nodes.end(), [&](const auto& node) {
            return node->getName() == name;
    });

    return (it == nodes.end())  ? nullptr : it->get();
}

Node* NN::variable(const Tensor& init_value)
{
    auto var = std::make_unique<Variable>(init_value);
    return graph.add_node(std::move(var));
}

Node* NN::constant(const Tensor& values)
{
    auto node = std::make_unique<Constant>(values);
    return graph.add_node(std::move(node));
}

Node* NN::placeholder(const Tensor::Shape& shape)
{
    auto node = std::make_unique<Placeholder>(shape);
    return graph.add_node(std::move(node));
}

Node* NN::placeholder(const Tensor::Shape& shape, const std::string& name)
{
    auto node = std::make_unique<Placeholder>(shape, name);
    return graph.add_node(std::move(node));
}

Node* NN::add(Node* n1, Node* n2)
{
    auto node = std::make_unique<Add>(n1, n2);
    return graph.add_node(std::move(node));
}

Node* NN::subtract(Node* n1, Node* n2)
{
    auto node = std::make_unique<Subtract>(n1, n2);
    return graph.add_node(std::move(node));
}

Node* NN::matmul(Node* n1, Node* n2)
{
    auto node = std::make_unique<MatMul>(n1, n2);
    return graph.add_node(std::move(node));
}

Node* NN::mul(Node* n1, Node* n2)
{
    auto node = std::make_unique<Mul>(n1, n2);
    return graph.add_node(std::move(node));
}

Node* NN::log(Node* n)
{
    auto node = std::make_unique<Log>(n);
    return graph.add_node(std::move(node));
}

Node* NN::pow2(Node* n)
{
    auto node = std::make_unique<Pow2>(n);
    return graph.add_node(std::move(node));
}

Node* NN::minus(Node* n)
{
    auto node = std::make_unique<Minus>(n);
    return graph.add_node(std::move(node));
}

Node* NN::sigmoid(Node* n)
{
    auto node = std::make_unique<Sigmoid>(n);
    return graph.add_node(std::move(node));
}

Node* NN::softmax(Node* n)
{
    auto node = std::make_unique<Softmax>(n);
    return graph.add_node(std::move(node));
}

