#include <iostream>
#include <vector>
#include <list>
#include <algorithm>
#include <queue>
#include <set>
#include <stack>
#include <chrono>
#include <functional>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <mpi.h>
#include<cstring>
#include<math.h>
#include <unordered_set>


class Graph {
private:
    int V;
    std::vector<std::list<int>> adjList;

    void bronKerbosch(std::unordered_set<int>& R, std::unordered_set<int>& P, std::unordered_set<int>& X, int& maxSize) const {
        if (P.empty() && X.empty()) {
            maxSize = std::max(maxSize, static_cast<int>(R.size()));
            return;
        }

        std::unordered_set<int> P_copy = P;

        for (int v : P_copy) {
            R.insert(v);
            std::unordered_set<int> P_new, X_new;

            for (int neighbor : adjList[v]) {
                if (P.count(neighbor)) P_new.insert(neighbor);
                if (X.count(neighbor)) X_new.insert(neighbor);
            }

            bronKerbosch(R, P_new, X_new, maxSize);

            R.erase(v);
            P.erase(v);
            X.insert(v);
        }
    }
public:
    Graph(int V) : V(V) {
        adjList.resize(V);
    }

    void add_edge(int u, int v) {
        adjList[u].push_back(v);
        adjList[v].push_back(u);
    }

    int degree(int v) const {
        return adjList[v].size();
    }

    int getV() const {
        return V;
    }

    const std::list<int>& get_neighbors(int v) const {
        return adjList[v];
    }
    int maximum_clique() const {
        std::unordered_set<int> R, P, X;
        for (int i = 0; i < V; i++) {
            P.insert(i);
        }

        int maxSize = 0;
        bronKerbosch(R, P, X, maxSize);
        return maxSize;
    }
    bool is_adjacent(int u, int v) const {
        return std::find(adjList[u].begin(), adjList[u].end(), v) != adjList[u].end();
    }

    // Function to compute the Welsh-Powell chromatic number bound
    int welshPowellChromaticNumber() {
        // Create a vector of pairs (vertex, degree)
        std::vector<std::pair<int, int>> vertices;
        for (int i = 0; i < V; ++i) {
            vertices.push_back({i, degree(i)});
        }

        // Sort vertices by degree in descending order
        std::sort(vertices.begin(), vertices.end(),
                  [this](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                      return a.second > b.second;
                  });

        // Calculate the Welsh-Powell bound
        int chromaticBound = 0;
        for (int i = 0; i < V; ++i) {
            int vertexDegree = vertices[i].second;
            int bound = std::min(vertexDegree + 1, i + 1); // i + 1 is the index in the sorted order (1-based)
            chromaticBound = std::max(chromaticBound, bound);
        }

        return chromaticBound;
    }

    std::pair<std::vector<int>, int> upperBound() const {
        std::vector<int> colors(V, -1);
        std::vector<int> saturation(V, 0);
        std::vector<int> degrees(V);

        // Initialize degrees
        for (int v = 0; v < V; ++v) {
            degrees[v] = adjList[v].size();
        }

        // Color all vertices
        for (int numColored = 0; numColored < V; ++numColored) {
            // Find uncolored vertex with maximum saturation degree
            int maxSaturation = -1;
            int maxDegree = -1;
            int selectedVertex = -1;

            for (int v = 0; v < V; ++v) {
                if (colors[v] == -1) {
                    if (saturation[v] > maxSaturation ||
                        (saturation[v] == maxSaturation && degrees[v] > maxDegree)) {
                        maxSaturation = saturation[v];
                        maxDegree = degrees[v];
                        selectedVertex = v;
                    }
                }
            }

            // Find the smallest available color for the selected vertex
            std::set<int> neighborColors;
            for (int neighbor : adjList[selectedVertex]) {
                if (colors[neighbor] != -1) {
                    neighborColors.insert(colors[neighbor]);
                }
            }

            //  smallest available color
            int color = 0;
            while (neighborColors.find(color) != neighborColors.end()) {
                color++;
            }
            colors[selectedVertex] = color;

            // updt saturation degrees of uncolored neighbors
            for (int neighbor : adjList[selectedVertex]) {
                if (colors[neighbor] == -1) {

                    std::set<int> distinctColors;
                    for (int neighborOfNeighbor : adjList[neighbor]) {
                        if (colors[neighborOfNeighbor] != -1) {
                            distinctColors.insert(colors[neighborOfNeighbor]);
                        }
                    }
                    saturation[neighbor] = distinctColors.size();
                }
            }
        }


        int numColors = 0;
        if (!colors.empty()) {
            numColors = *std::max_element(colors.begin(), colors.end()) + 1;
        }

        return {colors, numColors};
    }

    int maxClique() {
        int cliqueSize = 0;
        std::vector<bool> inClique(V, false);

        // sort
        std::vector<std::pair<int, int>> vbydeg;
        for (int i = 0; i < V; i++) {
            vbydeg.push_back({getDegree(i), i});
        }
        std::sort(vbydeg.begin(), vbydeg.end(),
                 [](const auto& a, const auto& b) { return a.first > b.first; });

        //from the highesrt degree vtx
        int firstVertex = vbydeg[0].second;
        cliqueSize = 1;
        inClique[firstVertex] = true;

        //add more vertices
        for (int i = 1; i < V; i++) {
            int v = vbydeg[i].second;
            bool canAdd = true;

            for (int j = 0; j < V; j++) {
                if (inClique[j] && !isAdjacent(v, j)) {
                    canAdd = false;
                    break;
                }
            }

            if (canAdd) {
                cliqueSize++;
                inClique[v] = true;
            }
        }

        return cliqueSize;
    }


private:

    bool isAdjacent(int u, int v) {
        for (int neighbor : adjList[u]) {
            if (neighbor == v) {
                return true;
            }
        }
        return false;
    }

    int getDegree(int v) const {
        return adjList[v].size();
    }
};

class Instance {
public:
    Graph& g;
    std::set<int> used_colors;
    int uncolored;
    std::vector<int> result;
    std::vector<int> saturation_degree;
    std::vector<std::set<int>> neighbor_colors;
    std::priority_queue<int, std::vector<int>, std::function<bool(int, int)>> pq;

    static bool cmp(int u, int v, const std::vector<int>& saturation_degree, const Graph& g) {
        if (saturation_degree[u] == saturation_degree[v]) {
            return g.degree(u) < g.degree(v);
        }
        return saturation_degree[u] < saturation_degree[v];
    }

    void update_saturation_degree(int vertex) {
        for (int neighbor : g.get_neighbors(vertex)) {
            if (result[neighbor] == -1) {
                int prev_saturation = saturation_degree[neighbor];
                saturation_degree[neighbor] = neighbor_colors[neighbor].size();
                if (prev_saturation != saturation_degree[neighbor]) {
                    pq.push(neighbor);
                }
            }
        }
    }

    int find_partner(int v1) {
        int v2;

        // First try: DSATUR heuristics
        if (!pq.empty()) {
            v2 = pq.top();
            pq.pop();

            // Check if v2 is uncolored and not adjacent to v1
            if (result[v2] == -1 && !g.is_adjacent(v1, v2)) {
                return v2;  // Return v2 if it's a valid partner
            } else {
                // If v2 is adjacent, push it back into the priority queue
                pq.push(v2);
            }
        }

        // Second try: first suitable one
        for (v2 = 0; v2 < g.getV(); ++v2) {
            if (result[v2] == -1 && !g.is_adjacent(v1, v2)) {
                return v2;
            }
        }

        return -1;
    }

    int color(int v) {
        std::vector<bool> available_colors(g.getV(), true);
        for (int neighbor : g.get_neighbors(v)) {
            if (result[neighbor] != -1) {
                available_colors[result[neighbor]] = false;
            }
        }
        int color = 0;
        while (!available_colors[color]) {
            ++color;
        }
        result[v] = color;
        used_colors.insert(color);
        neighbor_colors[v].insert(color);
        update_saturation_degree(v);
        uncolored--;
        return color;
    }

    int same_color(int v, int fixed_color) {
        std::vector<bool> available_colors(g.getV(), true);
        for (int neighbor : g.get_neighbors(v)) {
            if (result[neighbor] != -1) {
                available_colors[result[neighbor]] = false;
            }
        }

        int color = fixed_color;
        while (!available_colors[color]) {
            color = (color + 1) % g.getV();  // attribute the same color only if it gives a valid coloring
        }

        result[v] = color;
        used_colors.insert(color);
        neighbor_colors[v].insert(color);
        update_saturation_degree(v);
        uncolored--;
        return color;
    }

    int different_color(int v, int fixed_color) {
        std::vector<bool> available_colors(g.getV(), true);
        for (int neighbor : g.get_neighbors(v)) {
            if (result[neighbor] != -1) {
                available_colors[result[neighbor]] = false;
            }
        }

        int color = 0;
        while ((!available_colors[color]) || (color == fixed_color)) {
            ++color;
        }

        result[v] = color;
        used_colors.insert(color);
        neighbor_colors[v].insert(color);
        update_saturation_degree(v);
        uncolored--;
        return color;
    }

    std::vector<int> different_colors(int v, int fixed_color, int curr_ub) {
        std::vector<int> colors;
        if( v == -1 ) return colors;
        std::vector<bool> available_colors(g.getV(), true);
        for (int neighbor : g.get_neighbors(v)) {
            if (result[neighbor] != -1) {
                available_colors[result[neighbor]] = false;
            }
        }

        for (int i = 0; i < g.getV(); i++)
        {
            if(available_colors.at(i) && i != fixed_color && colors.size() < curr_ub)
                colors.push_back(i);
        }

        return colors;
    }

    Instance(Graph& g)
        : g(g), used_colors(), uncolored(g.getV()), result(g.getV(), -1),
          saturation_degree(g.getV(), 0), neighbor_colors(g.getV()),
          pq([this](int u, int v) { return cmp(u, v, saturation_degree, this->g); }) {
        for (int i = 0; i < g.getV(); ++i) {
            pq.push(i);
        }
    }
    Instance(const Instance& I, int v2, int color)
        : Instance(I)
        {
            result[v2] = color;
            used_colors.emplace(color);
            neighbor_colors[v2].insert(color);
            update_saturation_degree(v2);
            uncolored--;
        }

        Instance(const Instance& I, int v2, bool var)
        : Instance(I)
        {
            if( !var )
            {
                for( int i = 0; i < g.getV() ; i++)
                {
                    used_colors.emplace(i);
                }
            }
        }

    Instance(const Instance& I, const std::string& type)
        : Instance(I) {
        int v1;
        int v2;

        // DSATUR heuristics
        while (!pq.empty()) {
            v1 = pq.top();
            pq.pop();
            if (result[v1] == -1) break;
        }

        int assigned_color = color(v1);
        v2 = find_partner(v1);
        if (v2 != -1) {
            if (type == "same color") {
                same_color(v2, assigned_color);
            } else {
                different_color(v2, assigned_color);
            }
        }
    }


    Instance(const Instance& I, int v1, int v2, int assigned_color, const std::string& type)
        : Instance(I) {

        if (v2 != -1) {
            if (type == "same color") {
                same_color(v2, assigned_color);
            } else {
                different_color(v2, assigned_color);
            }
        }
    }

    int get_uncolored() const {
        return uncolored;
    }

    int get_colors() const {
        return used_colors.size();  // Return the number of unique colors (used_colors is a set);
    }

    // Serialize instance into a byte buffer
    std::vector<char> serialize() const {
        std::vector<char> buffer;
        int gV = g.getV();

        // Reserve space for fixed-size data
        buffer.resize(sizeof(int) * (3 + gV));  // uncolored, number of colors, result array

        int offset = 0;

        // Copy uncolored
        std::memcpy(buffer.data() + offset, &uncolored, sizeof(int));
        offset += sizeof(int);

        // Copy number of colors used
        int num_colors = used_colors.size();
        std::memcpy(buffer.data() + offset, &num_colors, sizeof(int));
        offset += sizeof(int);

        // Copy result vector
        std::memcpy(buffer.data() + offset, result.data(), gV * sizeof(int));
        offset += gV * sizeof(int);

        // Append used colors set
        for (int color : used_colors) {
            buffer.insert(buffer.end(), reinterpret_cast<char*>(&color), reinterpret_cast<char*>(&color) + sizeof(int));
        }

        return buffer;
    }

    // Deserialize an Instance from a byte buffer
    void deserialize(const std::vector<char>& buffer) {
        int offset = 0;

        // Read uncolored
        std::memcpy(&uncolored, buffer.data() + offset, sizeof(int));
        offset += sizeof(int);

        // Read number of colors used
        int num_colors;
        std::memcpy(&num_colors, buffer.data() + offset, sizeof(int));
        offset += sizeof(int);

        // Read result vector
        result.resize(g.getV());
        std::memcpy(result.data(), buffer.data() + offset, g.getV() * sizeof(int));
        offset += g.getV() * sizeof(int);

        // Read used colors set
        used_colors.clear();
        for (int i = 0; i < num_colors; i++) {
            int color;
            std::memcpy(&color, buffer.data() + offset, sizeof(int));
            offset += sizeof(int);
            used_colors.insert(color);
        }
    }
};

std::pair<std::vector<int>, std::pair<int,int>> algo(Graph& g, int rank, int size) {
    auto start_time = std::chrono::high_resolution_clock::now();
    int max_colors ;
    int lb = 0, ub = 0;
    int msize = 0;
    std::pair<std::vector<int>, int> result;
    std::vector<Instance> local_stack;
    std::vector<int> local_result;

    if (rank == 0) {
        lb = g.maxClique();
        result = g.upperBound();
        ub = result.second ;
        local_result = result.first;
        std::cout << "Initial LB: " << lb << std::endl;
        std::cout << "Initial UB: " << ub << std::endl;
    }

    MPI_Bcast(&lb, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ub, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (lb == ub) {
        std::cout << "Thread " << rank << " is returning " << ub << " " << lb << std::endl;
        std::pair<std::vector<int>, std::pair<int,int>> solution;
        solution.first = result.first;
        solution.second.first = result.second;
        solution.second.second = rank;
        return solution;
    }

    if (rank == 0) {
        std::vector<Instance> master_stack;
        Instance I(g);
        master_stack.push_back(I);
        int nc = g.getV();
        int c = 0;
        // Master explores left side of tree
        while ( g.getV() - nc <= 2 || master_stack.size() < size ) {
            Instance I = master_stack.back();
            master_stack.pop_back();
            nc = I.get_uncolored();

            if (I.get_colors() < ub) {
                if (nc == 0) {
                    ub = I.get_colors();
                    local_result = I.result;
                    std::cout << "Master rank break" << std::endl;
                    std::cout << "Upper bound value = " << ub << std::endl;
                    break;
                } else {
                    int v1;
                    int v2;
                    while (!I.pq.empty()) {
                        v1 = I.pq.top();
                        I.pq.pop();
                        if (I.result[v1] == -1) break;
                    }

                    int assigned_color = I.color(v1);
                    v2 = I.find_partner(v1);
                    std::vector<int> av_colors;
                    av_colors = I.different_colors(v2, assigned_color, ub);

                    Instance S(I, v1, v2, assigned_color, "same color");
                    master_stack.push_back(S);
                    int used = 0;
                    max_colors = av_colors.size();
                    for( int col : av_colors )
                    {
                        if (used == max_colors) break;
                        Instance D(I, v2, col);
                        master_stack.push_back(D);
                        used++;
                    }
                }
            }
        }
        std::cout << "Master rank break" << std::endl;
        while (  master_stack.size() < size-1)
        {
            Instance F(master_stack.back(),0,false);
            master_stack.push_back(F);
        }
        std::cout << "Master size " << master_stack.size() << std::endl;
        msize = master_stack.size();
        // Calculate work distribution
        int worker_count = size - 1;  // Excluding master
        std::vector<int> send_counts(size, 0);
        std::vector<int> instance_indices(size);  // To store which instances go to which worker

        // Calculate basic distribution - each worker gets approximately the same number of instances
        int base_count = msize / worker_count;
        int remainder = msize % worker_count;

        // Set up send counts for workers
        for (int i = 1; i < size; i++) {
            send_counts[i] = base_count;
            if (i <= remainder) {
                send_counts[i]++;
            }
        }

        // Create mapping of which instances go to which worker using interleaved pattern
        std::vector<std::vector<int>> worker_instances(size);

        // Implement interleaved distribution (front-to-back pairing)
        int front_idx = 0;
        int back_idx = msize - 1;
        int current_worker = 1;

        while (front_idx <= back_idx) {
            // Assign front instance to current worker
            if (worker_instances[current_worker].size() < send_counts[current_worker]) {
                worker_instances[current_worker].push_back(front_idx);
                front_idx++;
            }

            // If we haven't reached the middle and the worker still needs more work,
            // assign the back instance to the same worker
            if (front_idx <= back_idx && worker_instances[current_worker].size() < send_counts[current_worker]) {
                worker_instances[current_worker].push_back(back_idx);
                back_idx--;
            }

            // Move to next worker, cycling back to first worker if needed
            current_worker++;
            if (current_worker >= size) {
                current_worker = 1;
            }
        }

        // Serialize and distribute instances
        for (int i = 1; i < size; i++) {
            if (send_counts[i] > 0) {
                // Send number of instances this worker will receive
                int num_instances = worker_instances[i].size();
                MPI_Send(&num_instances, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

                // Pack and send instances for this worker
                for (int idx : worker_instances[i]) {
                    if (idx < master_stack.size()) {
                        std::vector<char> serialized_data = master_stack[idx].serialize();
                        int data_size = serialized_data.size();

                        // Send size of serialized data first
                        MPI_Send(&data_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                        // Send the actual serialized data
                        MPI_Send(serialized_data.data(), data_size, MPI_BYTE, i, 0, MPI_COMM_WORLD);
                    }
                }
            }
}
    }
    MPI_Bcast(&msize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if ( rank != 0 ){
        // Worker threads receive their instances
        MPI_Status status;

        // First receive the number of instances assigned to this worker
        int num_instances;
        MPI_Recv(&num_instances, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        // Receive instances
        for (int i = 0; i < num_instances; i++) {
            int data_size;
            MPI_Recv(&data_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

            std::vector<char> recv_buffer(data_size);
            MPI_Recv(recv_buffer.data(), data_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);

            Instance recv_instance(g);
            recv_instance.deserialize(recv_buffer);

            local_stack.push_back(recv_instance);
        }

        // Process local work

        
        while (!local_stack.empty()) {
            Instance I = local_stack.back();
            local_stack.pop_back();
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            //16.6 min
            if ( duration > 1000000 ) {break;}

            if (I.get_colors() < ub) {
                if (I.get_uncolored() == 0) {

                    ub = I.get_colors();
                    local_result = I.result;

                    if( ub == lb) {
                        break;
                    }

                } else {
                    int v1;
                    int v2;
                    while (!I.pq.empty()) {
                        v1 = I.pq.top();
                        I.pq.pop();
                        if (I.result[v1] == -1) break;
                    }
                    
                    int assigned_color = I.color(v1);
                    v2 = I.find_partner(v1);
                    if (v2 != -1) {
                    std::vector<int> av_colors;
                    av_colors = I.different_colors(v2, assigned_color, ub);

                    Instance S(I, v1, v2, assigned_color, "same color");
                    local_stack.push_back(S);
                    int used = 0;
                    //Some options
                    //max_colors = 2 + std::log2(rank);
                    //max_colors = (ub+lb)/lb + std::max( 3, std::log2(rank));

                    //this is the 100% accurate
                    max_colors = av_colors.size();
                    
                    for( int col : av_colors )
                    {
                        if (used == max_colors) break;
                        Instance D(I, v2, col);
                        local_stack.push_back(D);
                        used++;
                    }
                    }
                    else
                    {
                        Instance A(I, v1, true);
                        local_stack.push_back(A);
                    }
                }
            }
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);
    // Synchronize results
    int global_ub;
    MPI_Allreduce(&ub, &global_ub, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    // Find out which processes have the optimal solution
    int has_optimal_solution = (ub == global_ub) ? 1 : 0;
    std::vector<int> processes_with_solution(size);
    MPI_Allgather(&has_optimal_solution, 1, MPI_INT, processes_with_solution.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Find the lowest rank process that has the optimal solution
    int min_rank_with_solution = -1;
    for (int i = 0; i < size; i++) {
        if (processes_with_solution[i] == 1) {
            min_rank_with_solution = i;
            break;
        }
    }

    std::pair<std::vector<int>, std::pair<int,int>> solution;
    solution.second.second = -1;

    if (rank == min_rank_with_solution) {
        solution.first = local_result;
        solution.second.first = global_ub;
        solution.second.second = rank;
    }

    return solution;
}


//MAIN TO CHECK ONLY FOR ONE SPECIFIC FILE

namespace fs = std::filesystem;



int main(int argc, char* argv[]) {
    std::string directory = "../instances";

    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Collect all files and their sizes
    std::vector<std::pair<std::string, uintmax_t>> files;
    for (const auto& entry : fs::directory_iterator(directory)) {
        std::string file_path = entry.path().string();
        uintmax_t file_size = fs::file_size(entry.path());
        files.push_back({file_path, file_size});
    }

    // Sort files by size (smallest to largest)
    std::sort(files.begin(), files.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second; // Compare file sizes
        });

    // Process files in order of increasing size
    for (const auto& file : files) {
        std::string filename = file.first;
       
        {
                int rank, size;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                MPI_Comm_size(MPI_COMM_WORLD, &size);

                std::ifstream file(filename);
                if (!file) {
                    std::cerr << "Error: Unable to open file " << filename << std::endl;
                    return -1;
                }

                std::string line;
                int numVertices = 0, numEdges = 0;

                while (std::getline(file, line)) {
                    std::istringstream iss(line);
                    std::string token;
                    iss >> token;

                    // Ignore comments
                    if (token == "c") {
                        continue;
                    } else if (token == "p") {
                        std::string type;
                        iss >> type >> numVertices >> numEdges;
                        if (type != "edge") {
                            std::cerr << "Error: Unsupported graph type in " << filename << std::endl;
                            return -1;
                        }
                        break;
                    }
                }

                Graph g(numVertices);

                file.clear();
                file.seekg(0, std::ios::beg);

                while (std::getline(file, line)) {
                    std::istringstream iss(line);
                    std::string token;
                    iss >> token;

                    if (token == "e") {
                        int u, v;
                        iss >> u >> v;
                        g.add_edge(u - 1, v - 1);
                    }
                }
                
                file.close();
                if (rank == 0) {
                    std::cout << "File: " << filename <<  std::endl;
                }
                auto start_time = std::chrono::high_resolution_clock::now();
                std::pair<std::vector<int>, std::pair<int,int>> solution = algo(g, rank, size);
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();        
                std::vector<int> coloration = solution.first;
                std::pair<int,int> info = solution.second;

                // Synchronize all processes
                MPI_Barrier(MPI_COMM_WORLD);

                if (rank == info.second) {
                    int chromatic_color = 0;
                    std::set<int> colors;
                    for (auto color : coloration) {
                        colors.insert(color);
                    }
                
                    std::cout << " CN  :  " << colors.size() << " --> filename = " << filename << std::endl;
                    std::cout << " COLORING [ ";
                    for (auto val : coloration) {
                        std::cout << " " << val << " ";
                    }
                    std::cout << " ] ";
                    std::cout << " \nexec time : " << duration;
                    std::cout << "\n====================================================================================================\n" << std::flush;
                }



    }
    }

    MPI_Finalize();
    return 0;
}
