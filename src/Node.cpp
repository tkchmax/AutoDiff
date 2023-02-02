#include "Node.h"

unsigned int Constant::i = 0;
unsigned int Variable::i = 0;
unsigned int Placeholder::i = 0;

Node::Node(const Tensor& value, NodeType type, std::string name, const Node::Inputs& inputs)
{
    this->name = name;
    this->value = value;
    this->type = type;
    this->inputs = inputs;
}

Node::Node(const Node& other)
{
    *this = other;
}

void Node::operator=(const Node& rsh)
{
    inputs.clear();
    name = rsh.name;
    type = rsh.type;
    value = rsh.value;
    output = rsh.output;
    gradient = rsh.gradient;
}

Tensor Node::forward(const Node::Inputs& input)
{
   return output = value;
}

std::vector<Tensor> Node::backward(const Tensor& grad)
{
    return { Tensor(output.getShape(), 1) };
}

void Node::setGradient(const Tensor& grad)
{
    gradient = grad;
}

NodeType Node::getType() const
{
    return type;
}

std::string Node::getName() const
{
    return name;
}

const Node::Inputs& Node::getInputs() const
{
    return inputs;
}

const Tensor& Node::getOutput() const
{
    return output;
}

const Tensor& Node::getValue() const
{
    return value;
}

void Node::setValue(const Tensor& val)
{
    value = val;
}

void Node::setName(const std::string& name)
{
    this->name = name;
}

void Node::setInputs(const Inputs& inputs)
{
    this->inputs = inputs;
}

const Tensor& Node::getGradient() const
{
    return gradient;
}

std::unique_ptr<Node> Node::clone() const
{
    return std::unique_ptr<Node>(this->clone_impl());
}


bool Node::operator==(const Node& rsh)
{
    return name == rsh.name;
}

void Variable::assign(const Tensor& new_value)
{
    if (new_value.getShape() != value.getShape()) {
        throw std::invalid_argument("Shapes not equal");
    }
    
    value = new_value;
}

std::ostream& operator<<(std::ostream& out, const Node& node)
{
    out << "[";
    out << "value={";
    out << node.value;
    out << "} ";
    out << "name=";
    out << node.name;
    out << "]\n";
    return out;
}

void Placeholder::resize(const Tensor::Shape& new_shape)
{
    value = Tensor(new_shape);
}
