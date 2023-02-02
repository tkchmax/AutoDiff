#include "Graph.h"
#include <cassert>

Graph::Graph(const Graph& rsh)
{
    *this = rsh;
}

void Graph::operator=(const Graph& rsh)
{
    //clone each node
    for (const auto& node : rsh.nodes) {
        nodes.push_back(std::move(node->clone()));
    }

    //making links between just cloned nodes
    for (const std::unique_ptr<Node>& node : rsh.nodes) {
        Node::Inputs inputs;
        for (const Node* input : node->getInputs()) {
            Node* node = find(input->getName());
            assert(node != nullptr);
            inputs.push_back(node);
        }

        Node* current_root = find(node->getName());
        assert(current_root != nullptr);
        current_root->setInputs(inputs);
    }
}

Node* Graph::add_node(std::unique_ptr<Node> node)
{
    nodes.push_back(std::move(node));
    return nodes.back().get();
}

Node* Graph::find(const std::string& name) const
{
    auto it = std::find_if(nodes.begin(), nodes.end(), [&](const auto& node) {
        return node->getName() == name;
    });

    return (it != nodes.end()) ? it->get() : nullptr;
}

const std::vector<std::unique_ptr<Node>>& Graph::getNodes() const
{
    return nodes;
}

void Graph::reset()
{
    nodes.clear();
}
