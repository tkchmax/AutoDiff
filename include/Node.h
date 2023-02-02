#pragma once
#ifndef NODE_H_
#define NODE_H_

#include "Tensor.h"
#include <memory>

enum class NodeType {
    VARIABLE,
    CONST,
    PLACEHOLDER,
    OPERATOR
};

class Node {
public:
    using Inputs = std::vector<Node*>;

public:
    Node(
        const Tensor& value,
        NodeType type,
        std::string name,
        const Inputs& inputs = Inputs()
    );
    Node(const Node& other);
    void operator=(const Node& rsh);

    virtual Tensor forward(const Inputs& input = Inputs());
    virtual std::vector<Tensor> backward(const Tensor& grad);

    void setGradient(const Tensor& grad);
    NodeType getType() const;
    std::string getName() const;
    const Inputs& getInputs() const;
    const Tensor& getOutput() const;
    const Tensor& getValue() const;
    void setValue(const Tensor& val);
    void setName(const std::string& name);
    void setInputs(const Inputs& inputs);
    const Tensor& getGradient() const;
    void setInputGradient(int idx, const Tensor& grad) { inputs[idx]->gradient = grad; }

    std::unique_ptr<Node> clone() const;
    bool operator==(const Node& rsh);
    friend std::ostream& operator<<(std::ostream& out, const Node& node);

private:
    virtual Node* clone_impl() const { return new Node(*this); }

protected:
    std::string name;
    NodeType type;
    Tensor value;
    Inputs inputs;
    Tensor output;
    Tensor gradient;
};

class Constant : public Node{
public:
    Constant(const Tensor& value) : Node(value, NodeType::CONST, "Const:"+std::to_string(i++)){}
private:
    virtual Constant* clone_impl() const override { return new Constant(*this); }
    static unsigned int i;
};

class Variable : public Node {
public:
    Variable() : Node(Tensor(), NodeType::VARIABLE, "Var") {}
    Variable(const Tensor& init_value) : Node(init_value, NodeType::VARIABLE, "Var:"+std::to_string(i++)){}
    void assign(const Tensor& new_value);
private:
    virtual Variable* clone_impl() const override { return new Variable(*this); }
    static unsigned int i;
};

class Placeholder : public Node {
public:
    Placeholder(const Tensor::Shape& shape, const std::string& name) : Node(Tensor(shape), NodeType::PLACEHOLDER, name){}
    Placeholder(const Tensor::Shape& shape) : Placeholder(shape, "Plh:"+std::to_string(i++)){}
    void resize(const Tensor::Shape& new_shape);
private:
    virtual Placeholder* clone_impl() const override { return new Placeholder(*this); }
    static unsigned int i;
};

#endif


