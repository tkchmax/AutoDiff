#pragma once
#ifndef SESSION_H_
#define SESSION_H_

#include "Operation.h"
#include <unordered_map>

typedef std::unordered_map<const Node*, Tensor> FeedDict;

class Session
{
public:
    Session();
    Session(Node* root);
    Tensor run(Node* node, const FeedDict& feed_dict);
    Tensor run(const FeedDict& feed_dict);
    Tensor grad(Node* df, Node* dx);

private:
    void TopologySort(Node* node);
    std::vector<Node*> sorted_nodes;
    Node* root;
};

#endif
