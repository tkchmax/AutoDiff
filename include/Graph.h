#pragma once
#ifndef GRAPH_H_
#define GRAPH_H_

#include "Operation.h"

class Graph
{
public:
    Graph() {};
    Graph(const Graph& rsh);
    void operator=(const Graph& rsh); 
    Node* add_node(std::unique_ptr<Node> node);
    Node* find(const std::string& name) const;
    const std::vector<std::unique_ptr<Node>>& getNodes() const;
    void reset();
private:
    std::vector<std::unique_ptr<Node>> nodes;
};

#endif

