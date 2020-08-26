interface Element {

}

/**
 * Node represents a single, primitive calculation on graph.
 * Producing a new parameter / input, can also be thought of as an calculation.
 * A node does not necessarily mean there is a tensor, the output of a node can be many things.
 * A node accepts one or many ordered inputs, from other nodes.
 */
interface Node {
    id: string;
    inputs: string[];
    name: string[]
    op: string;
    attrs: Map<string, any>;
    parent: Cluster;
}

/**
 * A cluster is linked with a set of nodes/sub-clusters.
 * The inputs of a cluster is the union of all inputs in its set.
 */
interface Cluster {
    id: string;
    inputs: string[];
    parent: Cluster;
}

interface Edge {
    id: string;
}

interface Tag {
    id: string;
    associatedElements: string[];
}

interface Visualizer {

}

interface Frame {
    graphNodes: Node[];
    graphClusters: Cluster[];
    graphEdges: Edge[];
    tags: Tag[];
    visualizers: Map<Tag, Visualizer>;
}

interface VisualizationConfig {
    initialFrame: Frame;
    frames: Frame[];

}