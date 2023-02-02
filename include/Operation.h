#pragma once
#ifndef OPERATION_H_
#define OPERATION_H_

#include "Node.h"

class Operation : public Node {
public:
    Operation(Node* n, const std::string& name) : Node(Tensor(), NodeType::OPERATOR, name, {n}) {}
};

class BinaryOperation : public Node {
public:
    BinaryOperation(Node* n1, Node* n2, const std::string& name) : Node(Tensor(), NodeType::OPERATOR, name, {n1,n2}) {}
};

class Add : public BinaryOperation {
public:
    Add(Node* n1, Node* n2) : BinaryOperation(n1, n2, "AddOp:" + std::to_string(i++)) {}
    Tensor forward(const Node::Inputs& input) override;
    std::vector<Tensor> backward(const Tensor& grad) override;

private:
    virtual Add* clone_impl() const override { return new Add(*this); }
    static unsigned int i;
};

class Subtract : public BinaryOperation {
public:
    Subtract(Node* n1, Node* n2) : BinaryOperation(n1, n2, "SubtractOp:" + std::to_string(i++)) {}
    Tensor forward(const Node::Inputs& input) override;
    std::vector<Tensor> backward(const Tensor& grad) override;

private:
    virtual Subtract* clone_impl() const override { return new Subtract(*this); }
    static unsigned int i;
};

class MatMul : public BinaryOperation {
public:
    MatMul(Node* n1, Node* n2) : BinaryOperation(n1, n2, "MatMulOp:" + std::to_string(i++)) {}
    Tensor forward(const Node::Inputs& input) override;
    std::vector<Tensor> backward(const Tensor& grad) override;

private:
    virtual MatMul* clone_impl() const override { return new MatMul(*this); }
    static unsigned int i;
};

class Mul : public BinaryOperation {
public:
    Mul(Node* n1, Node* n2) : BinaryOperation(n1, n2, "MulOp:" + std::to_string(i++)) {}
    Tensor forward(const Node::Inputs& input) override;
    std::vector<Tensor> backward(const Tensor& grad) override;

private:
    virtual Mul* clone_impl() const override { return new Mul(*this); }
    static unsigned int i;
};

class Log : public Operation {
public:
    Log(Node* n) : Operation(n, "LogOp:" + std::to_string(i++)) {}
    Tensor forward(const Node::Inputs& input) override;
    std::vector<Tensor> backward(const Tensor& grad) override;

private:
    virtual Log* clone_impl() const override { return new Log(*this); }
    static unsigned int i;
};

class Pow2 : public Operation {
public:
    Pow2(Node* n) : Operation(n, "Pow_2Op:" + std::to_string(i++)) {}
    Tensor forward(const Node::Inputs& input) override;
    std::vector<Tensor> backward(const Tensor& grad) override;

private:
    virtual Pow2* clone_impl() const override { return new Pow2(*this); }
    static unsigned int i;
};

class Minus : public Operation {
public:
    Minus(Node* n) : Operation(n, "MinusOp:" + std::to_string(i++)) {}
    Tensor forward(const Node::Inputs& input) override;
    std::vector<Tensor> backward(const Tensor& grad) override;

private:
    virtual Minus* clone_impl() const override { return new Minus(*this); }
    static unsigned int i;
};

class Sigmoid : public Operation {
public:
    Sigmoid(Node* n) : Operation(n, "SigmoidOp:" + std::to_string(i++)) {}
    Tensor forward(const Node::Inputs& input) override;
    std::vector<Tensor> backward(const Tensor& grad) override;

private:
    virtual Sigmoid* clone_impl() const override { return new Sigmoid(*this); }
    static unsigned int i;
};

class Softmax : public Operation {
public:
    Softmax(Node* n) : Operation(n, "SoftmaxOp:" + std::to_string(i++)) {}
    Tensor forward(const Node::Inputs& input) override;
    std::vector<Tensor> backward(const Tensor& ground_truth) override;

private:
    virtual Softmax* clone_impl() const override { return new Softmax(*this); }
    static unsigned int i;
};

#endif