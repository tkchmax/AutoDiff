#include "Session.h"
#include <cassert>

namespace {
    void MakeOrdering(Node* node, std::vector<Node*>& order, std::vector<const Node*>& vis) {
        if (node->getType() == NodeType::OPERATOR) {
            for (Node* input : node->getInputs()) {
                if (std::find(vis.begin(), vis.end(), input) == vis.end()) {
                    MakeOrdering(input, order, vis);
                }
            }
        }

        vis.push_back(node);
        order.push_back(node);
    }
}

Session::Session()
{
    root = nullptr;
}

Session::Session(Node* root)
{
    this->root = root;
    TopologySort(root);
}

Tensor Session::run(Node* root, const FeedDict& feed_dict)
{
    this->root = root;
    TopologySort(root);
    return run(feed_dict);
}

Tensor Session::run(const FeedDict& feed_dict)
{
    for (Node* node : sorted_nodes) {
        if (node->getType() == NodeType::PLACEHOLDER) {
            if (feed_dict.at(node).getShape() != node->getValue().getShape()) {
                throw std::exception("Wrong feed value shape");
            }
            node->setValue(feed_dict.at(node));
        }
        node->forward(node->getInputs());
    }

    return root->getOutput();
}

Tensor Session::grad(Node* df, Node* dx)
{
    for (auto it = sorted_nodes.rbegin(); *it != dx; ++it) {
        if (it == sorted_nodes.rend()) {
            throw;
        }

        if ((*it)->getType() == NodeType::OPERATOR) {
            std::vector<Tensor> grads = (*it)->backward((*it)->getGradient());
            assert((*it)->getInputs().size() == grads.size());
            for (int i = 0; i < grads.size(); ++i) {
                (*it)->setInputGradient(i, grads[i]);
            }
        }
    }
    return dx->getGradient();
}

void Session::TopologySort(Node* node)
{
    sorted_nodes.clear();
    std::vector<const Node*> vis;
    MakeOrdering(node, sorted_nodes, vis);
}

