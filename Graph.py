import random
import math
import heapq
import time

import FibonacciHeap


class Graph:
    def __init__(self, size):
        self.adj_matrix = [[0]* size for _ in range (size)]
        self.size = size
        self.vertex_data = ['']* size # Store extra data for each vertex

    def add_edge (self, u, v, weight):
        if 0 <= u < self.size and 0 <= v < self.size:
            self.adj_matrix[u][v] = weight
            self.adj_matrix[v][u] = weight

    def add_vertex_data(self, vertex, data):
        """
        Associates additional data with a vertex.

        """
        if 0 <= vertex < self.size:
            self.vertex_data[vertex] = data


    # Graph generator:
    @staticmethod
    def generate_graph (number_nodes, density = 0.2):
        graph = Graph(number_nodes)
        max_number_edges = (number_nodes * (number_nodes-1))//2
        number_edges = int(max_number_edges*density)
        edges = set()
        while len(edges) < number_edges:
            u,v = random.sample(range(number_nodes),2)
            weight = random.randint(1,20)
            if (u,v) not in edges and (v,u) not in edges:
                graph.add_edge(u,v, weight)
                edges.add((u,v))

        return graph

# Dijkstra array based implementation
def dijkstra_graph_class(graph, source):
    number_nodes = graph.size
    if number_nodes==0 or source<0 or source>=number_nodes:
        return{}


    # Initialize all distances from nodes to source to infinity
    distance = {node:float('inf') for node in range(number_nodes)}
    distance[source] =0
    visited_nodes = [False]* number_nodes


    for _ in range(number_nodes):
        min_distance = float('inf')
        current_node = None

        # Find the unvisited node woth the smallest known distance
        for node in range (number_nodes):
            if not visited_nodes[node] and distance[node]< min_distance:
                min_distance = distance[node]
                current_node= node

        if current_node is None:# The end of nodes
            break

        visited_nodes[current_node]= True

        # Update distance to adjacent nodes:

        for neighbor  in range(number_nodes) :
          weight = graph.adj_matrix[current_node][neighbor]
          if weight > 0 and not visited_nodes[neighbor]:
              new_distance = distance[current_node] + weight
              if new_distance< distance[neighbor]:
                   distance[neighbor]= new_distance
    return distance


# Dijkstra binary heap implementation
def shortest_weight_graph_path(graph, source):
    number_nodes = graph.size
    if source<0 or source>=number_nodes:
        return  None


    # Create and initialize a dictionary distance that containing the elements of the graph
    distance ={node:math.inf for node in range(number_nodes)}
    distance[source] =0 # Distance to the source node is 0
    visited_vertices = set() # Set to track visited nodes
    heap =[(0, source)] # Min_heap priority queue for the Dijkstra's algorithm
    heapq.heappush(heap,(0,source)) # Push the source with distance 0

    while heap:
        current_distance, node = heapq.heappop(heap) # Pop the node with the  smallest distance
        if node in visited_vertices:
            continue
        visited_vertices.add(node) # Mark the node as visited

        # Explore all neighbors (connected nodes)
        for neighbor in range(number_nodes):
            weight = graph.adj_matrix[node][neighbor]
            if weight > 0 and neighbor not in visited_vertices:
                new_distance = current_distance +weight
                if new_distance<distance[neighbor]:
                    distance[neighbor]= new_distance
                    heapq.heappush(heap,(new_distance, neighbor))

    return distance


def dijkstra_fibonacci_heap(graph,source):
    number_nodes = graph.size
    fib_heap = FibonacciHeap()
    if source<0 or source>=number_nodes:
        return  None

    distance ={node: math.inf for node in range(number_nodes)}
    distance[source]=0

    fibonacci_heap_nodes = {}
 # Insert all nodes into Fibonacci heap and store them in  fibonacci heap:
    for node in range(number_nodes):
        fib_node = fib_heap.insert(distance[node], node)
        fibonacci_heap_nodes[node]= fib_node

    while fib_heap:
        min_node = fib_heap.extract_min() # Extract the node with minimum distance
        node = min_node[1]
        current_distance = min_node[0]







# Generate a random graph with 100 nodes and density 0.1
graph = Graph.generate_graph(100, density=0.1)


# Test Dijkstra with basic algorithm
execution_time_array_based = time.time()
result_dijkstra_distances = dijkstra_graph_class(graph, source=0)
end_time = time.time()
print(f"Basic Dijkstra runtime: {end_time - execution_time_array_based:.6f} seconds")

# Test Dijkstra with heap-based algorithm
execution_time_binary_heap = time.time()
heap_based_distance = shortest_weight_graph_path(graph, source=0)
end_time_binary_heap = time.time()
print(f"Heap-based Dijkstra runtime: {end_time_binary_heap - execution_time_binary_heap:.6f} seconds")



