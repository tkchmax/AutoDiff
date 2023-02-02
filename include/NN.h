#pragma once
#ifndef NN_H_
#define NN_H_

#include "Graph.h"

class NN
{
public:
    NN();
    static NN& ByDefault();
    void reset();
    Node* findByName(const std::string& name) const;
    Node* variable(const Tensor& init_value);
    Node* constant(const Tensor& values);
    Node* placeholder(const Tensor::Shape& shape);
    Node* placeholder(const Tensor::Shape& shape, const std::string& name);
    Node* add(Node* n1, Node* n2);
    Node* subtract(Node* n1, Node* n2);
    Node* matmul(Node* n1, Node* n2);
    Node* mul(Node* n1, Node* n2);
    Node* log(Node* n);
    Node* pow2(Node* n);
    Node* minus(Node* n);
    Node* sigmoid(Node* n);
    Node* softmax(Node* n);
private:
    Graph graph;
};

#endif
