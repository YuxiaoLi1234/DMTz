#include <iostream>
#include <float.h> 
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <stdio.h>
#include <parallel/algorithm>  
#include <unordered_map>
#include <random>
#include <iostream>
#include <limits.h>
#include <cstring> 
#include <chrono> 
#include <cuda_runtime.h>
#include <string>
#include <omp.h>
#include <unordered_set>
#include <set>
#include <map>
#include <algorithm>
#include <numeric>
#include <utility>
#include <iomanip>
#include <bitset>
#include <chrono>
#include <filesystem>
using namespace std;
namespace fs = std::filesystem;
// nvcc -c Msc_3d.cu -o MSC
struct Node {
    int x, y;  
    struct Node* next;
};

using std::count;
struct Vertex;
struct Edge;
struct Face;
struct Tetrahedra;
__device__ int numNeighbors = 14;
__device__ int directions1[14][3] = 
{{1,0,0},{-1,0,0},
{0,1,0},{0,-1,0},
{0,0,1}, {0,0,-1},
{-1,1,0},{1,-1,0}, 
{0,1,1},{0,-1,-1},  
{-1,0,1},{1,0,-1},
{-1,1,1}, {1,-1,-1}};


__device__ double* decp_data;
__device__ double* input_data;
__device__ Vertex* vertices;
__device__ Vertex* dec_vertices;
__device__ Edge* edges;
__device__ Edge* dec_edges;
__device__ Face* faces;

__device__ Face* dec_faces;
__device__ Tetrahedra* tetrahedras;
__device__ Tetrahedra* dec_tetrahedras;



__device__ int count_f_max;
__device__ int count_f_min;
__device__ int count_f_saddle;
__device__ int count_f_2_saddle;
__device__ int count_f_as_vpath;
__device__ int count_f_ds_vpath;
__device__ int count_f_ds_wall;
__device__ int count_f_as_wall;
__device__ int* paired_vertices;
__device__ int* preserve_type1;
__device__ int* paired_faces;
__device__ int* paired_edges;
__device__ int* paired_tetrahedras;
__device__ int* connected_max_saddle;
__device__ int* connected_min_saddle;

__device__ int* delta_counter;
__device__ int* neighbors_index;

__device__ int* visited_faces;
__device__ int simplification;
__device__ int* unchangeable_vertices;

__device__ int* dec_paired_vertices;
__device__ int* dec_paired_edges;
__device__ int* dec_paired_faces;
__device__ int* dec_paired_tetrahedras;

__device__ int* vertex_paired_edge;
__device__ int* edge_paired_vertex;
__device__ int* edge_paired_face;
__device__ int* face_paired_edge;
__device__ int* face_paired_tetra;
__device__ int* tetra_paired_face;

__device__ double range;
__device__ int num_Vertices;
__device__ int num_Edges;
__device__ int num_Faces;
__device__ int num_Tetrahedras;
__device__ int edit_type;


__device__ double bound;
__device__ double persistence;
__device__ double dec_persistence;
__device__ double* d_deltaBuffer;
__device__ int* changed_index;
__device__ int* false_max; 
__device__ int* false_min;
__device__ int* false_saddle;
__device__ int* false_2saddle;
__device__ int* false_as_vpath;
__device__ int* false_ds_vpath;
__device__ int* false_ds_wall;
__device__ int* false_as_wall;
__device__ int* false_as_wall_index;
__device__ int* false_ds_wall_index;
__device__ int* false_as_vpath_index;
__device__ int* false_ds_vpath_index;

__device__ int num;
using namespace std;
double bound1;

double epsilon = 1e-12;
int width,height,depth,size2;
std::string inputfilename;
std::string cpfilename;
int preserve_extrema;
int preserve_saddles;
int preserve_vpath;
int preserve_geometry;
int simplification1;
int filtration;
int preserve_connectors;
int preserve_type = 0;
int edit_type1;
double range1;
double maxValue, minValue, compression_time, additional_time;

// std::set<int> mySet;
// std::vector<int> f_saddles;
// std::vector<int> PL_type;
// 定义顶点结构
std::vector<double> getdata2(std::string filename){
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    std::vector<double> data;
    if (!file) {
        std::cerr << "can not open file" << std::endl;
        return data;
    }

    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    
    std::streamsize num_floats = size / sizeof(double);
    

    std::vector<double> buffer(num_floats);

    
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        
        return buffer;
    } else {
        std::cerr << "can not read file" << std::endl;
        return buffer;
    }
    
    return buffer;
}

struct Vertex {
    int x, y,z;
    double value;
    int id;
    int dim;
    int e_persistence[14] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    int number_of_s = 0;
    int neighbors[14] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    int edges[14] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    bool paired;
    int filtered = 0;
    int paired_edge_id = -1;

    __host__ __device__ Vertex() : x(0), y(0), z(0), value(0.0), id(0), dim(0), paired(false), paired_edge_id(-1) {}
    __host__ __device__ Vertex(int x, int y, int z, double value, int id) : x(x), y(y), z(z), value(value), id(id), dim(0), paired(false), paired_edge_id(-1) {}
    
};




struct Edge 
{
    int v1;
    int v2;
    double weight;
    int id = -1;
    bool paired = false;
    int paired_face_id = -1;
    int paired_vertex_id=-1;
    int isValid = 0;
    int visited = -1;
    int faces[10] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    int v_persistence[2];
    int filtered = 0;
    double largest_value = -1;
    __host__ __device__ Edge() : v1(-1), v2(-1), weight(0.0), id(-1), paired(false), paired_face_id(-1) {
    }
    __host__ __device__ Edge(int v1, int v2, double weight) : v1(v1), v2(v2), weight(weight), id(-1), paired(false), paired_face_id(-1) {
        int Sorted_vertex1 = v1;
        int Sorted_vertex2 = v2;
        if (v1 < v2) {
            Sorted_vertex1 = v2;
            Sorted_vertex2 = v1;
        }
        this->v1 = Sorted_vertex1;
        this->v2 = Sorted_vertex2; 
        isValid = 1;
        // largest_value = v1->value>v2->value?v1->value:v2->value;
        
        
    }
    
};


__device__ double get_delta(int id)
{
    if(edit_type==1) return -pow(2.0, -delta_counter[id]) * bound;
    else return (input_data[id]-bound-decp_data[id])/2.0;
    
}

__device__ int getTriangleID1(int v1, int v2, int v3, int width, int height, int depth, int direction = 0) 
{
    int id = -1;
    int sortedV1 = v1;
    int sortedV2 = v2;
    int sortedV3 = v3;


    if (sortedV1 < sortedV2) { int temp = sortedV1; sortedV1 = sortedV2; sortedV2 = temp; }
    if (sortedV1 < sortedV3) { int temp = sortedV1; sortedV1 = sortedV3; sortedV3 = temp; }
    if (sortedV2 < sortedV3) { int temp = sortedV2; sortedV2 = sortedV3; sortedV3 = temp; }

    
    
    int min_x = min(v1%width, min(v2%width, v3%width));
    int min_y = min((v1/width)%height, min((v2/width)%height, (v3/width)%height));
    int min_z = min((v1 / (width * height)) % depth, min((v2 / (width * height)) % depth, (v3 / (width * height)) % depth));
    
    // xy palne
    int sortedV1_x = sortedV1%width;
    int sortedV1_y = (sortedV1/width) % height;
    int sortedV1_z = (sortedV1/ (width * height)) % depth;

    int sortedV2_x = sortedV2%width;
    int sortedV2_y = (sortedV2/width) % height;
    int sortedV2_z = (sortedV2/ (width * height)) % depth;

    int sortedV3_x = sortedV3 % width;
    int sortedV3_y = (sortedV3 /width) % height;
    int sortedV3_z = (sortedV3 / (width * height)) % depth;
    if(direction == 0){

        int cell_id = min_y * (width - 1) + min_x;
        if(sortedV3_x == sortedV1_x && sortedV1_x == min_x && sortedV3_y == sortedV2_y && sortedV2_y == min_y)
        {
            
            return cell_id * 2 + 2 * min_z * (width - 1) * (height - 1);
        }

        else
        {
            return cell_id * 2 + 1 + 2 * min_z * (width - 1) * (height - 1);
        }
    }
    // yz plane
    else if(direction == 1){
        int cell_id = min_y * (depth - 1) + min_z;
        if(sortedV2_y == sortedV1_y && sortedV1_y == min_y + 1 && sortedV3_z == sortedV2_z && sortedV2_z == min_z)
        {
            
            return cell_id * 2 + 2 * depth  * (width - 1) * (height - 1) + 2 * min_x * (height - 1) * (depth - 1);
        }
        else
        {
            // printf("%d\n",cell_id * 2 + 1 + 2 * depth * (width - 1) * (height - 1) + 2 * min_x * (height - 1) * (depth - 1));
            
            
            return cell_id * 2 + 1 + 2 * depth * (width - 1) * (height - 1) + 2 * min_x * (height - 1) * (depth - 1);
        }
    }
    //  xz plane
    // if(sortedV1_x == 9 && sortedV1_y == 9 && sortedV1_z == 9) printf("v1: %d %d %d, v2: %d %d %d, v3:%d %d %d\n", sortedV1_x, sortedV1_y, sortedV1_z, sortedV2_x, sortedV2_y, sortedV2_z, sortedV3_x, sortedV3_y, sortedV3_z);
    else if(direction == 2){
        int cell_id = min_z * (width - 1) + min_x;
        // if(sortedV1_x == 9 && sortedV1_y == 9 && sortedV1_z == 9) printf("v1: %d %d %d, v2: %d %d %d, v3:%d %d %d\n", sortedV1_x, sortedV1_y, sortedV1_z, sortedV2_x, sortedV2_y, sortedV2_z, sortedV3_x, sortedV3_y, sortedV3_z);
        if(sortedV2_z == sortedV3_z && sortedV2_z == min_z && sortedV1_x == sortedV3_x && sortedV1_x == min_x)
        {
            id = cell_id * 2 + 2* (depth * (width - 1) * (height - 1) + width * (height - 1) * (depth - 1) + min_y*(width-1)*(depth-1));
            // if(id == 4858) printf("v1: %d %d %d, v2: %d %d %d, v3:%d %d %d\n", sortedV1_x, sortedV1_y, sortedV1_z, sortedV2_x, sortedV2_y, sortedV2_z, sortedV3_x, sortedV3_y, sortedV3_z);
            
            return id;
        }
        else
        {
            
            // printf("%d\n", cell_id * 2 + 1 + 2*(depth * (width - 1) * (height - 1) + width * (height - 1) * (depth - 1)  + min_y*(width-1)*(depth-1)));
            return cell_id * 2 + 1 + 2* (depth * (width - 1) * (height - 1) + width * (height - 1) * (depth - 1) + min_y*(width-1)*(depth-1));
        }

    }

    else if(direction == 3){
        int cell_id = min_z * (width - 1) * (height - 1) + min_y * (width - 1) + min_x;
        
        int current_id = -1;
        if(sortedV3_x == min_x && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x + 1&& sortedV2_y == min_y && sortedV2_z == min_z &&
           sortedV1_x == min_x && sortedV1_y== min_y+1 && sortedV1_z == min_z +1 
        ) {current_id = 0;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x && sortedV2_y == min_y + 1 && sortedV2_z == min_z &&
           sortedV1_x == min_x && sortedV1_y== min_y + 1 && sortedV1_z == min_z+1 
        ) {current_id = 1;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x + 1 && sortedV2_y == min_y + 1 && sortedV2_z == min_z &&
           sortedV1_x == min_x && sortedV1_y== min_y+1 && sortedV1_z == min_z +1 
        ) {current_id = 2;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x && sortedV2_y == min_y && sortedV2_z == min_z + 1 &&
           sortedV1_x == min_x && sortedV1_y== min_y+1 && sortedV1_z == min_z +1 
        ) {current_id = 3;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x + 1 && sortedV2_y == min_y && sortedV2_z == min_z + 1 &&
           sortedV1_x == min_x && sortedV1_y== min_y+1 && sortedV1_z == min_z +1 
        ) {current_id = 4;}
        if(sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
           sortedV2_x == min_x && sortedV2_y == min_y + 1 && sortedV2_z == min_z + 1 &&
           sortedV1_x == min_x + 1 && sortedV1_y== min_y + 1 && sortedV1_z == min_z +1 
        ) {current_id = 5;}
        id = cell_id * 6 + current_id + 2 * depth * (width - 1) * (height - 1) + 2 * width * (height - 1) * (depth - 1) +  2 * height * (width - 1) * (depth - 1);
        // printf("%d\n", id);
        return id;
        
    }
    return -1;
    

}
__device__ int getTriangleID(int v1, int v2, int v3, int width, int height, int depth)
{
    // xy plane

    int sortedV1 = v1;
    int sortedV2 = v2;
    int sortedV3 = v3;


    if (sortedV1 < sortedV2) { int temp = sortedV1; sortedV1 = sortedV2; sortedV2 = temp; }
    if (sortedV1 < sortedV3) { int temp = sortedV1; sortedV1 = sortedV3; sortedV3 = temp; }
    if (sortedV2 < sortedV3) { int temp = sortedV2; sortedV2 = sortedV3; sortedV3 = temp; }

    int sortedV1_x = sortedV1%width;
    int sortedV1_y = (sortedV1/width) % height;
    int sortedV1_z = (sortedV1/ (width * height)) % depth;

    int sortedV2_x = sortedV2%width;
    int sortedV2_y = (sortedV2/width) % height;
    int sortedV2_z = (sortedV2/ (width * height)) % depth;

    int sortedV3_x = sortedV3 % width;
    int sortedV3_y = (sortedV3 /width) % height;
    int sortedV3_z = (sortedV3 / (width * height)) % depth;
    

    if(sortedV1_z == sortedV2_z && sortedV2_z == sortedV3_z)
    {
        int id = getTriangleID1(sortedV1, sortedV2, sortedV3, width, height, depth) ;
        
        return id;
    }
    // yz plane
    if(sortedV1_x == sortedV2_x && sortedV2_x == sortedV3_x)
    {
        int id = getTriangleID1(sortedV1, sortedV2, sortedV3, width, height, depth, 1) ;
        // printf("sortedV1: %d %d %d, sortedV2: %d %d %d, sortedV3: %d %d %d, id: %d \n", sortedV1_x, sortedV1_y, sortedV1_z, sortedV2_x, sortedV2_y, sortedV2_z, sortedV3_x, sortedV3_y, sortedV3_z, id + sortedV1_x * 2*(width-1)*(height-1) + depth * 2*(width-1)*(height-1) );
        return id;
    }

    
    
    if(sortedV1_y == sortedV2_y && sortedV2_y == sortedV3_y)
    {
        // if(sortedV1_x == 9 && sortedV1_y == 9 && sortedV1_z == 9) printf("v1: %d %d %d, v2: %d %d %d, v3:%d %d %d\n", sortedV1_x, sortedV1_y, sortedV1_z, sortedV2_x, sortedV2_y, sortedV2_z, sortedV3_x, sortedV3_y, sortedV3_z);
        int id = getTriangleID1(sortedV1, sortedV2, sortedV3, width, height, depth, 2);
        // printf("sortedV1: %d %d %d, sortedV2: %d %d %d, sortedV3: %d %d %d, id: %d \n", sortedV1_x, sortedV1_y, sortedV1_z, sortedV2_x, sortedV2_y, sortedV2_z, sortedV3_x, sortedV3_y, sortedV3_z, id + sortedV1_y * 2*(width-1)*(height-1) + depth * 2*(width-1)*(height-1) + width * 2*(width-1)*(height-1) );
        return id;
    }
    
    else
    {
        int id = getTriangleID1(sortedV1, sortedV2, sortedV3, width, height, depth, 3);
        // printf("sortedV1: %d %d %d, sortedV2: %d %d %d, sortedV3: %d %d %d, id: %d \n", sortedV1_x, sortedV1_y, sortedV1_z, sortedV2_x, sortedV2_y, sortedV2_z, sortedV3_x, sortedV3_y, sortedV3_z, id + sortedV1_y * 2*(width-1)*(height-1) + depth * 2*(width-1)*(height-1) + width * 2*(width-1)*(height-1) );
        return id;

    }

    return -1;
}

__device__ void getVerticesFromTriangleID(
    int triangleID, 
    int width, 
    int height, 
    int depth, 
    int &v1, 
    int &v2, 
    int &v3) {

    int baseID, min_x, min_y, min_z;
    int starID = triangleID;
    // if(starID == 212468) printf("type1:\n");
    // XY 平面的三角形
    
    if (triangleID < 2 * depth * (width - 1) * (height - 1)) {
        baseID = triangleID / 2;
        min_z = baseID / ((width - 1) * (height - 1));
        min_y = (baseID % ((width - 1) * (height - 1))) / (width - 1);
        min_x = baseID % (width - 1);

        // 三角形类型（上下三角形）
        if (triangleID % 2 == 0) {
            v1 = min_x + min_y * width + min_z * width * height;
            v2 = (min_x + 1) + min_y * width + min_z * width * height;
            v3 = min_x + (min_y + 1) * width + min_z * width * height;
        } else {
            v1 = (min_x + 1) + (min_y + 1) * width + min_z * width * height;
            v2 = (min_x + 1) + min_y * width + min_z * width * height;
            v3 = min_x + (min_y + 1) * width + min_z * width * height;
        }

        if (v1 < v2) { int temp = v1; v1 = v2; v2 = temp; }
        if (v1 < v3) { int temp = v1; v1 = v3; v3 = temp; }
        if (v2 < v3) { int temp = v2; v2 = v3; v3 = temp; }
        // if(starID == 2945) printf("type1: %d %d %d\n", v1, v2, v3);
        return;
        
    }

    // YZ 平面的三角形
    triangleID -= 2 * depth * (width - 1) * (height - 1);
    if (triangleID < 2 * width * (height - 1) * (depth - 1)) {
        baseID = triangleID / 2;
        min_x = baseID / ((height - 1) * (depth - 1));
        min_y = (baseID % ((height - 1) * (depth - 1))) / (depth - 1);
        min_z = baseID % (depth - 1);
        
        if (triangleID % 2 == 0) {
            v1 = min_x + min_y * width + min_z * width * height;
            v2 = min_x + (min_y + 1) * width + min_z * width * height;
            v3 = min_x + (min_y + 1) * width + (min_z + 1) * width * height;
        } else {
            v1 = min_x + (min_y) * width + (min_z) * width * height;
            v2 = min_x + (min_y) * width + (min_z + 1) * width * height;
            v3 = min_x + (min_y+1) * width + (min_z + 1) * width * height;
        }
        if (v1 < v2) { int temp = v1; v1 = v2; v2 = temp; }
        if (v1 < v3) { int temp = v1; v1 = v3; v3 = temp; }
        if (v2 < v3) { int temp = v2; v2 = v3; v3 = temp; }
        // if(starID == 2945) printf("type2: %d %d %d %d %d\n", v1, v2, v3, starID, triangleID);
        return;
    }

    // XZ 平面的三角形
    triangleID -= 2 * width * (height - 1) * (depth - 1);
    if (triangleID < 2 * height * (width - 1) * (depth - 1)) {
        baseID = triangleID / 2;
        min_y = baseID / ((width - 1) * (depth - 1));
        min_z = (baseID % ((width - 1) * (depth - 1))) / (width - 1);
        min_x = baseID % (width - 1);

        if (triangleID % 2 == 0) {
            v1 = min_x + min_y * width + min_z * width * height;
            v2 = (min_x + 1) + min_y * width + min_z * width * height;
            v3 = min_x + min_y * width + (min_z + 1) * width * height;
        } else {
            v1 = (min_x + 1) + min_y * width + (min_z) * width * height;
            v2 = (min_x) + min_y * width + (min_z + 1) * width * height;
            v3 = (min_x+1) + min_y * width + (min_z + 1) * width * height;
        }
        if (v1 < v2) { int temp = v1; v1 = v2; v2 = temp; }
        if (v1 < v3) { int temp = v1; v1 = v3; v3 = temp; }
        if (v2 < v3) { int temp = v2; v2 = v3; v3 = temp; }
        // if(starID == 2945) printf("type3: %d %d %d\n", v1, v2, v3, starID);
        return;
    }

    // 对角线方向的三角形
    triangleID -= 2 * height * (width - 1) * (depth - 1);
    baseID = triangleID / 6;
    min_z = baseID / ((width - 1) * (height - 1));
    min_y = (baseID % ((width - 1) * (height - 1))) / (width - 1);
    min_x = baseID % (width - 1);
    // int y2 = triangleID / ((width - 1) * (depth - 1));
    // int z2 = (triangleID % ((width - 1) * (depth - 1))) / (depth - 1);
    // int x2 = triangleID % (depth - 1);

    int subTriangleType = triangleID % 6;
    
    switch (subTriangleType) {
        
        case 0:
            v1 = min_x + (min_y) * width + (min_z) * width * height;
            v2 = min_x+1 + min_y * width + min_z * width * height;
            v3 = (min_x ) + (min_y +1)* width + (min_z+1) * width * height;
            break;
        case 1:
            v1 = (min_x +1) + (min_y ) * width + (min_z) * width * height;
            v2 = min_x + (min_y + 1) * width + min_z * width * height;
            v3 = min_x + (min_y + 1) * width + (min_z+1)* width * height;
            break;
        case 2:
            v1 = min_x + 1 + min_y * width + min_z * width * height;
            v2 = (min_x + 1) + (min_y + 1) * width + min_z * width * height;
            v3 = (min_x) + (min_y +1)* width + (min_z +1) * width * height;
            break;
        case 3:
            v1 = (min_x + 1) + min_y * width + (min_z) * width * height;
            v2 = (min_x) + min_y * width + ( min_z+1) * width * height;
            v3 = min_x + (min_y + 1) * width + (min_z + 1) * width * height;
            break;
        case 4:
            v1 = (min_x + 1) + min_y * width + (min_z) * width * height;
            v2 = (min_x + 1) + (min_y) * width +( min_z +1 ) * width * height;
            v3 = (min_x) + (min_y + 1) * width + (min_z + 1)* width * height;
            break;
        case 5:
            v1 = (min_x + 1) + min_y * width + (min_z) * width * height;
            v2 = (min_x) + (min_y + 1) * width + (min_z + 1) * width * height;
            v3 = (min_x + 1) + (min_y + 1) * width +( min_z+1) * width * height;
            break;
    }

    // if(starID == 2945) printf("type4: %d %d %d %d %d %d\n", v1, v2, v3, starID, triangleID, subTriangleType);
    if (v1 < v2) { int temp = v1; v1 = v2; v2 = temp; }
    if (v1 < v3) { int temp = v1; v1 = v3; v3 = temp; }
    if (v2 < v3) { int temp = v2; v2 = v3; v3 = temp; }
    
    return;
    
}






__device__ int getTetrahedraID(int v1, int v2, int v3, int v4, int width, int height, int depth) {
    // 找到包含这四个顶点的立方体的位置

    int cellX = min(min(v1%width, v2%width), min(v3%width, v4%width));
    int cellY = min(min((v1/width)%height, (v2/width)%height), min((v3/width)%height, (v4/width)%height));
    int cellZ = min(min((v1 / (width * height)) % depth, (v2 / (width * height)) % depth), min((v3 / (width * height)) % depth, (v4 / (width * height)) % depth));

    int min_x = cellX;
    int min_y = cellY;
    int min_z = cellZ;

    
    int cellID = cellZ * (width - 1) * (height - 1) + cellY * (width - 1) + cellX;
    
    int value1 = v1;
    int value2 = v2;
    int value3 = v3;
    int value4 = v4;

    
    int sortedV1 = v1;
    int sortedV2 = v2;
    int sortedV3 = v3;
    int sortedV4 = v4;

    
    if (sortedV1 < sortedV2) { int temp = sortedV1; sortedV1 = sortedV2; sortedV2 = temp; }
    if (sortedV1 < sortedV3) { int temp = sortedV1; sortedV1 = sortedV3; sortedV3 = temp; }
    if (sortedV1 < sortedV4) { int temp = sortedV1; sortedV1 = sortedV4; sortedV4 = temp; }

    if (sortedV2 < sortedV3) { int temp = sortedV2; sortedV2 = sortedV3; sortedV3 = temp; }
    if (sortedV2 < sortedV4) { int temp = sortedV2; sortedV2 = sortedV4; sortedV4 = temp; }

    if (sortedV3 < sortedV4) { int temp = sortedV3; sortedV3 = sortedV4; sortedV4 = temp; }

    int sortedV1_x = sortedV1%width;
    int sortedV1_y = (sortedV1/width) % height;
    int sortedV1_z = (sortedV1/ (width * height)) % depth;

    int sortedV2_x = sortedV2%width;
    int sortedV2_y = (sortedV2/width) % height;
    int sortedV2_z = (sortedV2/ (width * height)) % depth;

    int sortedV3_x = sortedV3 % width;
    int sortedV3_y = (sortedV3 /width) % height;
    int sortedV3_z = (sortedV3 / (width * height)) % depth;
    
    int sortedV4_x = sortedV4 % width;
    int sortedV4_y = (sortedV4 /width) % height;
    int sortedV4_z = (sortedV4 / (width * height)) % depth;

    if (sortedV4_x == min_x && sortedV4_y == min_y && sortedV4_z == min_z &&
        sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
        sortedV2_x == min_x && sortedV2_y == min_y + 1 && sortedV2_z == min_z &&
        sortedV1_x == min_x && sortedV1_y == min_y + 1 && sortedV1_z == min_z+1
        ) {
        return cellID * 6; // 0 1 2 6
    } 
    if (sortedV4_x == min_x && sortedV4_y == min_y && sortedV4_z == min_z &&
        sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z &&
        sortedV2_x == min_x && sortedV2_y == min_y && sortedV2_z == min_z + 1 &&
        sortedV1_x == min_x && sortedV1_y == min_y + 1 && sortedV1_z == min_z + 1
        ) {
        return cellID * 6 + 1; // 0 1 4 6
    } 
    if (sortedV4_x == min_x + 1 && sortedV4_y == min_y && sortedV4_z == min_z &&
        sortedV3_x == min_x && sortedV3_y == min_y + 1 && sortedV3_z == min_z &&
        sortedV2_x == min_x + 1 && sortedV2_y == min_y + 1 && sortedV2_z == min_z &&
        sortedV1_x == min_x && sortedV1_y == min_y + 1 && sortedV1_z == min_z + 1
        ){
        return cellID * 6 + 2; // 1 2 6 3
    } 
    if (sortedV4_x == min_x + 1 && sortedV4_y == min_y && sortedV4_z == min_z &&
        sortedV3_x == min_x + 1 && sortedV3_y == min_y + 1 && sortedV3_z == min_z &&
        sortedV2_x == min_x && sortedV2_y == min_y + 1 && sortedV2_z == min_z+1 &&
        sortedV1_x == min_x + 1&& sortedV1_y == min_y +1  && sortedV1_z == min_z + 1
        ) {
        return cellID * 6 + 3; //  1 3 6 7
    } 
    if (sortedV4_x == min_x + 1 && sortedV4_y == min_y && sortedV4_z == min_z &&
        sortedV3_x == min_x && sortedV3_y == min_y && sortedV3_z == min_z + 1 &&
        sortedV2_x == min_x + 1 && sortedV2_y == min_y && sortedV2_z == min_z+1 &&
        sortedV1_x == min_x && sortedV1_y == min_y + 1 && sortedV1_z == min_z + 1
        ) {
        return cellID * 6 + 4; // 1 4 6 5
    } 
    if (sortedV4_x == min_x + 1 && sortedV4_y == min_y && sortedV4_z == min_z &&
        sortedV3_x == min_x + 1 && sortedV3_y == min_y && sortedV3_z == min_z + 1 &&
        sortedV2_x == min_x && sortedV2_y == min_y + 1 && sortedV2_z == min_z+1 &&
        sortedV1_x == min_x + 1 && sortedV1_y == min_y + 1 && sortedV1_z == min_z + 1
        ) {
        
        return cellID * 6 + 5; // 1 5 6 7
    }

    
    // printf("%d %d %d %d\n", sortedV4->id, sortedV3->id, sortedV2->id, sortedV1->id);
    return -1;
}

__device__ void getVerticesFromTetrahedraID(
    int tetrahedraID, 
    int width, 
    int height, 
    int depth, 
    int &v1, 
    int &v2, 
    int &v3, 
    int &v4) {

    // 计算四面体所在单元格的 ID 和类型
    int baseID = tetrahedraID / 6;
    int type = tetrahedraID % 6;

    // 计算单元格位置
    int cellZ = baseID / ((width - 1) * (height - 1));
    int cellY = (baseID % ((width - 1) * (height - 1))) / (width - 1);
    int cellX = baseID % (width - 1);

    
    switch (type) {
        case 0:
            v1 = cellX + (cellY) * width + (cellZ) * width * height;
            v2 = cellX + 1 + cellY * width + cellZ * width * height;
            v3 = (cellX ) + (cellY + 1) * width + cellZ * width * height;
            v4 = cellX + (cellY + 1) * width + (cellZ +1) * width * height;
            break;
        case 1:
            v1 = cellX + (cellY) * width + (cellZ) * width * height;
            v2 = (cellX + 1) + cellY * width + cellZ * width * height;
            v3 = cellX + cellY * width + (cellZ + 1) * width * height;
            v4 = cellX + (cellY + 1) * width + (cellZ +1) * width * height;
            break;
        case 2:
            v1 = (cellX + 1) + cellY * width + cellZ * width * height;
            v2 = cellX + (cellY + 1) * width + cellZ * width * height;
            v3 = (cellX + 1) + (cellY + 1) * width + cellZ * width * height;
            v4 = cellX + (cellY + 1) * width + (cellZ + 1) * width * height;
            break;
        case 3:
            v1 = (cellX + 1) + cellY * width + cellZ * width * height;
            v2 = (cellX + 1) + (cellY + 1) * width + cellZ * width * height;
            v3 = cellX + (cellY + 1) * width + (cellZ + 1) * width * height;
            v4 = (cellX + 1) + (cellY + 1) * width + (cellZ + 1) * width * height;
            break;
        case 4:
            v1 = (cellX + 1) + cellY * width + cellZ * width * height;
            v2 = cellX + cellY * width + (cellZ + 1) * width * height;
            v3 = (cellX + 1) + cellY * width + (cellZ + 1) * width * height;
            v4 = cellX + (cellY + 1) * width + (cellZ + 1) * width * height;
            break;
        case 5:
            v1 = (cellX + 1) + cellY * width + cellZ * width * height;
            v2 = (cellX + 1) + cellY * width + (cellZ + 1) * width * height;
            v3 = cellX + (cellY + 1) * width + (cellZ + 1) * width * height;
            v4 = (cellX + 1) + (cellY + 1) * width + (cellZ + 1) * width * height;
            break;
    }


    if (v1 < v2) { int temp = v1; v1 = v2; v2 = temp; }
    if (v1 < v3) { int temp = v1; v1 = v3; v3 = temp; }
    if (v1 < v4) { int temp = v1; v1 = v4; v4 = temp; }

    if (v2 < v3) { int temp = v2; v2 = v3; v3 = temp; }
    if (v2 < v4) { int temp = v2; v2 = v4; v4 = temp; }

    if (v3 < v4) { int temp = v3; v3 = v4; v4 = temp; }
}



// int calculateTetrahedronID(int i, int j, int k, int t, int width, int height, int depth) {
//     return k * (width - 1) * (height - 1) * 6 + j * (width - 1) * 6 + i * 6 + t;
// }
// class SimpleStack {
// public:
//     __device__ SimpleStack(int initialSize) : capacity(initialSize), size(0) {
//         data = (int*)malloc(capacity * sizeof(int));
//     }
//     __device__ ~SimpleStack() {
//         free(data);
//     }

//     __device__ void clear(int newCapacity) {
//         // 检查是否需要释放
//         if (data != nullptr) {
//             free(data);  // 释放原来的内存
//             data = nullptr;  // 设置指针为 nullptr 防止悬挂指针
//         }

//         capacity = newCapacity;  // 设置新的容量
//         size = 0;  // 重置大小为0

//         // 重新分配内存，并检查是否成功
//         data = (int*)malloc(capacity * sizeof(int));
//         if (data == nullptr) {
//             // 处理内存分配失败的情况
//             printf("Memory allocation failed\n");
//             return;
//         }
//     }

//     __device__ void clear1() {
        
//         if (data) {
//             free(data); 
//             data = nullptr;
//         }
//     }

//     __device__ int getElement(int index) const {
//         if (index >= 0 && index < size) {
//             return data[index];
//         } else {
//             printf("Index out of bounds!\n");
//             return -1;  // 返回特殊值表示下标越界
//         }
//     }
//     // 压栈
//     __device__ void push(int value) {
//         if (size >= capacity) {
//             // 扩展容量
//             int newCapacity = capacity * 2;
//             int* newData = (int*)malloc(newCapacity * sizeof(int));
//             for (int i = 0; i < size; ++i) {
//                 newData[i] = data[i];
//             }
//             free(data);
//             data = newData;
//             capacity = newCapacity;
//         }
//         data[size] = value;
//         size++;
//     }

//     // 出栈
//     __device__ int pop() {
//         if (!isEmpty()) {
//             size--;
//             return data[size];
//         } else {
//             printf("Stack is empty!\n");
//             return -1; // 返回特殊值表示栈为空
//         }
//     }

//     // 获取栈的大小
//     __device__ int getSize() const {
//         return size;
//     }

//     // 判断栈是否为空
//     __device__ bool isEmpty() const {
//         return size == 0;
//     }

// private:
//     int* data;
//     int capacity;
//     int size;
// };

class SimpleStack {
public:
    __device__ SimpleStack(int* sharedMemory, int sharedCapacity, int* globalMemory, int globalCapacity) 
        : data(sharedMemory), capacity(sharedCapacity), size(0), useGlobal(false), 
          globalData(globalMemory), globalCapacity(globalCapacity) {}

    __device__ ~SimpleStack() {
        // 由于shared memory是CUDA管理的，因此不需要手动释放
        // 只释放在global memory中申请的内存
        if (useGlobal && globalData) {
            free(globalData);
        }
    }

    // Push operation
    __device__ void push(int value) {
        if (size >= capacity) {
            // Shared memory is full, switch to global memory
            if (!useGlobal) {
                useGlobal = true;
                for (int i = 0; i < size; ++i) {
                    globalData[i] = data[i];
                }
            } else if (size >= globalCapacity) {
                // Expand the global memory if needed
                int newCapacity = globalCapacity * 2;
                int* newData = (int*)malloc(newCapacity * sizeof(int));
                if (newData == nullptr) {
                    printf("Global memory allocation failed\n");
                    return;
                }
                for (int i = 0; i < size; ++i) {
                    newData[i] = globalData[i];
                }
                free(globalData);
                globalData = newData;
                globalCapacity = newCapacity;
            }
            // Store the value in global memory
            globalData[size] = value;
        } else {
            // Store the value in shared memory
            data[size] = value;
        }
        size++;
    }

    // Pop operation
    __device__ int pop() {
        if (!isEmpty()) {
            size--;
            if (useGlobal) {
                return globalData[size];
            } else {
                return data[size];
            }
        } else {
            printf("Stack is empty!\n");
            return -1;  // Return special value indicating empty stack
        }
    }

    // Get stack size
    __device__ int getSize() const {
        return size;
    }

    // Check if stack is empty
    __device__ bool isEmpty() const {
        return size == 0;
    }

private:
    int* data;           // Pointer to shared memory
    int* globalData;     // Pointer to global memory
    int capacity;        // Capacity of shared memory
    int globalCapacity;  // Capacity of global memory
    int size;            // Current stack size
    bool useGlobal;      // Flag indicating whether global memory is used
};


struct Face 
{
    int v1;
    int v2;
    int v3;
    double value;
    int id;
    bool paired;
    int paired_edge_id;
    int visited = -1;
    int paired_tetrahedra_id;
    int tetra[2] = {-1, -1};
    int filtered = 0;
    int t_persistence[2] = {-1, -1};
    int isValid = 0;

    double largest_value = -1;
    __host__ __device__ Face() : v1(-1), v2(-1), v3(-1), value(0.0), id(-1), paired(false), paired_edge_id(-1), paired_tetrahedra_id(-1) {}
    __host__ __device__ Face(int v1, int v2, int v3) : v1(v1), v2(v2), v3(v3), value(-1), id(-1), paired(false), paired_edge_id(-1), paired_tetrahedra_id(-1) {
        int value1 = v1;
        int value2 = v2;
        int value3 = v3;

        int sortedV1 = v1;
        int sortedV2 = v2;
        int sortedV3 = v3;


        if (sortedV1 < sortedV2) { int temp = sortedV1; sortedV1 = sortedV2; sortedV2 = temp; }
        if (sortedV1 < sortedV3) { int temp = sortedV1; sortedV1 = sortedV3; sortedV3 = temp; }
        if (sortedV2 < sortedV3) { int temp = sortedV2; sortedV2 = sortedV3; sortedV3 = temp; }
        this->v1 = sortedV1;
        this->v2 = sortedV2;
        this->v3 = sortedV3;
        
        isValid = 1;
        
        
        
    }
};

struct Tetrahedra
{
    int v1;
    int v2;
    int v3;
    int v4;

    double value;
    int id;
    bool paired;
    int paired_face_id;
    int isValid = 0;
    int f_persistence[2];
    double largest_value = -1;
    int filtered = 0;
    __host__ __device__ Tetrahedra() : v1(-1), v2(-1), v3(-1), v4(-1), value(0.0), id(-1), paired(false), paired_face_id(-1) {}
    __host__ __device__ Tetrahedra(int v1, int v2, int v3, int v4) : v1(v1), v2(v2), v3(v3), v4(v4), value(-1), id(-1), paired(false), paired_face_id(-1) {
        int value1 = v1;
        int value2 = v2;
        int value3 = v3;
        int value4 = v4;

        
        int sortedV1 = v1;
        int sortedV2 = v2;
        int sortedV3 = v3;
        int sortedV4 = v4;

        if (sortedV1 < sortedV2) { int temp = sortedV1; sortedV1 = sortedV2; sortedV2 = temp; }
        if (sortedV1 < sortedV3) { int temp = sortedV1; sortedV1 = sortedV3; sortedV3 = temp; }
        if (sortedV1 < sortedV4) { int temp = sortedV1; sortedV1 = sortedV4; sortedV4 = temp; }
        if (sortedV2 < sortedV3) { int temp = sortedV2; sortedV2 = sortedV3; sortedV3 = temp; }
        if (sortedV2 < sortedV4) { int temp = sortedV2; sortedV2 = sortedV4; sortedV4 = temp; }
        if (sortedV3 < sortedV4) { int temp = sortedV3; sortedV3 = sortedV4; sortedV4 = temp; }

        this->v1 = sortedV1;
        this->v2 = sortedV2;
        this->v3 = sortedV3;
        this->v4 = sortedV4;
        
        isValid = 1;

       
    }
};





__device__ int getEdgeID(int x, int y, int z, int x1, int y1, int z1, int width, int height, int depth) {
    int dx = x1 - x;
    int dy = y1 - y;
    int dz = z1 - z;
    int id ;
    if (dx == 1 && dy == 0 && dz == 0) {  
        // printf("x axis: %d\n", z * (width - 1) * height + y * (width - 1) + x);
        return z * (width - 1) * height + y * (width - 1) + x;
    } else if (dx == -1 && dy == 0 && dz == 0) {  
        int id = z * (width - 1) * height + y * (width - 1) + x - 1;
        // printf("x axis: %d\n", id);
        return z * (width - 1) * height + y * (width - 1) + (x - 1);
    } else if (dx == 0 && dy == 1 && dz == 0) {  // Y 方向的正向边
        // printf("y axis: %d\n", (width - 1) * height * depth + z * width * (height - 1) + x * (height - 1) + y);
         return (width - 1) * height * depth + z * width * (height - 1) + x * (height - 1) + y;
     } else if (dx == 0 && dy == -1 && dz == 0) {  // Y 方向的负向边
    //  printf("y axis: %d\n", (width - 1) * height * depth 
        // + z * width * (height - 1) 
        // + x * (height - 1) 
        // + y - 1);
        return (width - 1) * height * depth 
        + z * width * (height - 1) 
        + x * (height - 1) 
        + y - 1;
    } else if (dx == 0 && dy == 0 && dz == 1) {  // Z 方向的正向边
    // printf("z axis: %d\n", (width - 1) * height * depth + width * (height - 1) * depth + y * width * (depth - 1) + x * (width - 1) + z);
        return (width - 1) * height * depth + width * (height - 1) * depth + y * width * (depth - 1) + x * (depth - 1) + z;
    } else if (dx == 0 && dy == 0 && dz == -1) {  // Z 方向的负向边
    // printf("z axis: %d\n", (width - 1) * height * depth + width * (height - 1) * depth + y * width * (depth - 1) + x * (width - 1) + z - 1);
        return (width - 1) * height * depth + width * (height - 1) * depth + y * width * (depth - 1) + x * (depth - 1) + z - 1;
    } else if (dx == 1 && dy == -1 && dz == 0) {  
        id =  (width - 1) * height * depth + width * (height - 1) * depth + width * height * (depth - 1)
            + z * (height - 1) * (width - 1) 
            + (y - 1) * (width - 1) 
            + x ;
        return id;
    } else if (dx == -1 && dy == 1 && dz == 0) {   
        id = (width - 1) * height * depth + width * (height - 1) * depth + width * height * (depth - 1)
             + z * (height - 1) * (width - 1) 
             + (y ) * (width - 1) 
             + (x - 1);
         
        return id;
    } else if (dx == 0 && dy == 1 && dz == 1) {  
        id =  (width - 1) * height * depth + width * (height - 1) * depth + width * height * (depth - 1)
             + (width - 1) * (height - 1) * depth 
             + x * (height - 1) * (depth - 1)
             + (y ) * (depth - 1) 
             + z ;
        
        return id ;
    } else if (dx == 0 && dy == -1 && dz == -1) {  
        id =  (width - 1) * height * depth + width * (height - 1) * depth + width * height * (depth - 1)
             + (width - 1) * (height - 1) * depth
             + x * (height - 1) * (depth - 1)
             + (y - 1) * (depth - 1) 
             + z - 1;
        
        return id;
    } else if (dx == -1 && dy == 0 && dz == 1) {   
        id = (width - 1) * height * depth + width * (height - 1) * depth + width * height * (depth - 1)
             + (width - 1) * (height - 1) * depth  + width * (height -1 )* (depth - 1) 
             + y * (width - 1) * (depth - 1) 
             + (z) * (width - 1) 
             + x - 1;
        

        return id;
    } else if (dx == 1 && dy == 0 && dz == -1) {  
        id = (width - 1) * height * depth + width * (height - 1) * depth + width * height * (depth - 1)
             + (width - 1) * (height - 1) * depth + width * (height -1 )* (depth - 1) 
             + y * (width - 1) * (depth - 1) 
             + (z - 1) * (width - 1) 
             + x;
        
        return id;
    }  else if (dx != 0 && dy != 0 && dz != 0) {
        int cellX = min(x, x1);
        int cellY = min(y, y1);
        int cellZ = min(z, z1);
        int cellid = cellZ * (width - 1) * (height - 1) + cellY * (width - 1) + cellX;
        id = (width - 1) * height * depth 
            + width * (height - 1) * depth 
            + width * height * (depth - 1)
            + (width - 1) * (height - 1) * (depth)
            + (width - 1) * (height) * (depth - 1)
            + (width ) * (height - 1) * (depth -1 )
            + cellid;
        
        return id;
    }
    
    

    return -1;  // 非法方向
}




__device__ void sortVerticesByData(int v1, int v2, int v3, int &sortedV1, int &sortedV2, int &sortedV3, int type = 0) {
    // 初始化 sortedV1, sortedV2, sortedV3
    double *data = input_data;
    if(type == 1) data = decp_data;
    sortedV1 = v1;
    sortedV2 = v2;
    sortedV3 = v3;

    // 比较 data 值并按要求排序，如果 data 值相等，则按 v1, v2, v3 大小排序
    if (data[sortedV1] < data[sortedV2] || (data[sortedV1] == data[sortedV2] && sortedV1 < sortedV2)) {
        int temp = sortedV1;
        sortedV1 = sortedV2;
        sortedV2 = temp;
    }
    if (data[sortedV1] < data[sortedV3] || (data[sortedV1] == data[sortedV3] && sortedV1 < sortedV3)) {
        int temp = sortedV1;
        sortedV1 = sortedV3;
        sortedV3 = temp;
    }
    if (data[sortedV2] < data[sortedV3] || (data[sortedV2] == data[sortedV3] && sortedV2 < sortedV3)) {
        int temp = sortedV2;
        sortedV2 = sortedV3;
        sortedV3 = temp;
    }
}

__device__ bool compare_vertices(int v1, int v2, int type)
{
    double *data = input_data;
    if(type == 1) data = decp_data;

    return data[v1] > data[v2] || (data[v1] == data[v2] && v1 > v2);
}

// __device__ bool compare_faces(int v1, int v2, int v3, int v1_1, int v2_1, int v3_1, int type)
// {
//     double *data = input_data;
//     if(type == 1) data = decp_data;

    
// }

__global__ void initializeKernel1() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<num_Vertices){
        d_deltaBuffer[tid] = -2000.0 * range;
        // changed_index[tid] = -1;
    }

}

__device__ double atomicCASDouble(double* address, double val) {
   
    uint64_t* address_as_ull = (uint64_t*)address;
    
    uint64_t old_val_as_ull = *address_as_ull;
    uint64_t new_val_as_ull = __double_as_longlong(val);
    uint64_t assumed;


    assumed = old_val_as_ull;
    
    // return atomicCAS((unsigned long long int*)address, (unsigned long long int)compare, (unsigned long long int)val);
    
    old_val_as_ull = atomicCAS((unsigned long long int*)address_as_ull, (unsigned long long int)assumed, (unsigned long long int)new_val_as_ull);
    // } while (assumed != old_val_as_ull);

    
    return __longlong_as_double(old_val_as_ull);
}

__device__ int swap1(int index, double diff)
{
    int update_successful = 0;
    double old_value = d_deltaBuffer[index];
    while (update_successful==0) {
        double current_value = d_deltaBuffer[index];
        if (diff > current_value) {
            double swapped = atomicCASDouble(&d_deltaBuffer[index], diff);
            if (swapped == current_value) {
                update_successful = 1;
                
            } else {
                old_value = swapped;
            }
        } else {
            update_successful = 1; 
    }
    }
    
}

__device__ int swap(int index, double diff)
{
    if(edit_type == 0)
    {
        double old_value = d_deltaBuffer[index];
        if ( diff > old_value) {                    
            swap1(index, diff);
        } 
    }
    
    else d_deltaBuffer[index] = diff;
}

__global__ void applyDeltaBuffer1(int width, int height, int depth) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_Vertices) {
        
        if(d_deltaBuffer[tid]!=-2000.0 * range){
            
            // printf("%d: %.17f %d\n", tid, d_deltaBuffer[tid], delta_counter[tid]);
            if(preserve_type1==0)
            {
                if(abs(d_deltaBuffer[tid])>1e-15 * range && delta_counter[tid]<5 && abs(input_data[tid]-(decp_data[tid] + d_deltaBuffer[tid])) <= bound){
                
                    decp_data[tid] += d_deltaBuffer[tid]; 
                    delta_counter[tid]+=1;
                    
                }
                else
                {
                    delta_counter[tid] = 6;
                    decp_data[tid] = input_data[tid] - bound;

                    // for (int i = 0; i < 14; i++) {
                    //     int neighbor_id = neighbors_index[tid*14+i];
                    //     if(neighbor_id==-1) continue;
                        
                    //     delta_counter[neighbor_id] = 10;
                    //     decp_data[neighbor_id] = input_data[neighbor_id] - bound;
                    // }
                    
                }
            }
            else
            {
                delta_counter[tid] = 6;
                decp_data[tid] = input_data[tid] - bound;
            }
            
        }
        
    }
    
}


__global__ void build_faces(int width, int height, int depth, int numVertices, int numEdges, int numFaces, int type=0, int a2 = 0) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numEdges) return;
        
    
    Edge* edge = &edges[index];
    Edge* dec_edge = &dec_edges[index];
    
    Vertex* v1 = &vertices[edge->v1];
    Vertex* v2 = &vertices[edge->v2];
    
    
    int M_alpha[10] = {-1,-1, -1, -1, -1, -1, -1, -1, -1, -1};
    int M_alpha_size = 0;

    
    int face_num = 0;
    for (int i = 0; i < numNeighbors; i++) {
        for (int j = 0; j < numNeighbors; j++) {
            int neighbor_id = v1->neighbors[i];
            int neighbor_id1 = v2->neighbors[j];
            
            if(neighbor_id == -1 or neighbor_id1 == -1) continue;
            if (neighbor_id == neighbor_id1) {
                
                
                Vertex *v3 = &vertices[neighbor_id];
                Face face(edge->v1, edge->v2, neighbor_id);
                Face dec_face(edge->v1, edge->v2, neighbor_id);
                
                int face_id = getTriangleID(face.v1, face.v2, face.v3, width, height, depth);
                
                face.id = face_id;
                dec_face.id = face_id;
                face.isValid = 1;

                int* isValidPtr = reinterpret_cast<int*>(&faces[face_id].isValid);
                if (atomicCAS(isValidPtr, 0, 1) == 0) {
                    faces[face_id] = face;
                   
                }

                int* isValidPtr1 = reinterpret_cast<int*>(&dec_faces[face_id].isValid);
                if (atomicCAS(isValidPtr1, 0, 1) == 0) {
                    
                    dec_faces[face_id] = dec_face;
                    
                }
                
                
                edge->faces[face_num] = face_id;
                dec_edge->faces[face_num] = face_id;
                face_num++;
             
            }
        }
    }
    
    
}



__global__ void build_tetras(int width, int height, int depth, int numVertices, int numEdges, int numFaces, int numTetrahedra, int type = 0, int a2 = 0) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numFaces) return;
    // printf("%d %d\n", numFaces, num_Tetrahedras);
    
    

    Face* face = &faces[index];
    Face* dec_face = &dec_faces[index];
    
    Vertex* v1 = &vertices[face->v1];
    Vertex* v2 = &vertices[face->v2];
    Vertex* v3 = &vertices[face->v3];

    int M_alpha[2] = {-1, -1};
    int M_alpha_size = 0;
    int face_num = 0;
    for (int i = 0; i < numNeighbors; i++) {
        for (int j = 0; j < numNeighbors; j++) {
            for (int k = 0; k < numNeighbors; k++) {
        
            int neighbor_id = v1->neighbors[i];
            int neighbor_id1 = v2->neighbors[j];
            int neighbor_id2 = v3->neighbors[k];
            if(neighbor_id == -1 or neighbor_id1 == -1 or neighbor_id2 == -1) continue;
            if(!(neighbor_id == neighbor_id1 && neighbor_id == neighbor_id2 && neighbor_id1 == neighbor_id2 )) continue;
            if (neighbor_id == v1->id || neighbor_id == v2->id || neighbor_id == v3->id || neighbor_id == -1) continue;

            Vertex* v4 = &vertices[neighbor_id];

            if(v1->x == v2->x and v1->x == v3->x and v1->x == v4->x) continue;
            if(v1->y == v2->y and v1->y == v3->y and v1->y == v4->y) continue;
            if(v1->z == v2->z and v1->z == v3->z and v1->z == v4->z) continue;
            if (v4->id == v2->id || v4->id == v3->id || v4->id == v1->id) continue;
            
            Tetrahedra tetra(face->v1, face->v2, face->v3, neighbor_id);
            Tetrahedra dec_tetra(face->v1, face->v2, face->v3, neighbor_id);

            int tetra_id = getTetrahedraID(tetra.v1, tetra.v2, tetra.v3, tetra.v4, width, height, depth);
            
            tetra.id = tetra_id;
            tetra.isValid = 1;

            dec_tetra.id = tetra_id;
            dec_tetra.isValid = 1;

            
            int* isValidPtr = reinterpret_cast<int*>(&tetrahedras[tetra_id].isValid);
            if (atomicCAS(isValidPtr, 0, 1) == 0) {
                
                tetrahedras[tetra_id] = tetra;
                
            }

            int* isValidPtr1 = reinterpret_cast<int*>(&dec_tetrahedras[tetra_id].isValid);
            if (atomicCAS(isValidPtr1, 0, 1) == 0) {
                dec_tetrahedras[tetra_id] = dec_tetra;
                
            }
            
            // if(tetra_id != face->tetra[1] && tetra_id != face->tetra[0] && face_num<2 )
            // {
            
            face->tetra[face_num] = tetra_id;
            dec_face->tetra[face_num] = tetra_id;
            face_num++;
            // }
            

    
    }}}
    
    
}

void checkCudaError(cudaError_t err, const char* msg) 
{
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void saveVectorToBin(const std::vector<double>& vec, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    size_t size = vec.size();
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));
    outFile.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(double));

    outFile.close();
}




double calculateMSE(const std::vector<double>& original, const std::vector<double>& compressed) {
    if (original.size() != compressed.size()) {
        throw std::invalid_argument("The size of the two vectors must be the same.");
    }

    double mse = 0.0;
    for (size_t i = 0; i < original.size(); i++) {
        mse += std::pow(static_cast<double>(original[i]) - compressed[i], 2);
    }
    mse /= original.size();
    return mse;
}

double calculatePSNR(const std::vector<double>& original, const std::vector<double>& compressed, double maxValue) {
    double mse = calculateMSE(original, compressed);
    if (mse == 0) {
        return std::numeric_limits<double>::infinity(); // Perfect match
    }
    double psnr = -20.0*log10(sqrt(mse)/maxValue);
    return psnr;
}

void writeDataToFile(const std::vector<unsigned long long>& data, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary | std::ios::out);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(unsigned long long));
        file.close();
    } else {
        std::cerr << "无法打开文件 " << filename << " 进行写入。" << std::endl;
    }
}


bool compressFile(const std::string& filename) {
    std::string command = "zstd -f " + filename;
    std::cout << "Executing command: " << command << std::endl;
    int result = std::system(command.c_str());
    return result == 0;
}

void writeBitmapToFile(const std::vector<uint8_t>& bitmap, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary | std::ios::out);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(bitmap.data()), bitmap.size());
        file.close();
    } else {
        std::cerr << "can not open file " << filename << std::endl;
    }
}

std::tuple<unsigned long long, unsigned long long> convertToIEEE754(double num) {
    
    unsigned long long* p = reinterpret_cast<unsigned long long*>(&num);

    
    int sign = (*p >> 63) & 1;

    
    unsigned long long exponent = (*p >> 52) & 0x7FF;

    
    unsigned long long mantissa = (*p) & 0xFFFFFFFFFFFFF;

    return std::make_tuple(exponent, mantissa);
}


std::vector<std::pair<int, int>> runLengthEncodeBitmap(const std::vector<uint8_t>& bitmap) {
    std::vector<std::pair<int, int>> rleData; // 存储 (位值, 数量) 对
    if (bitmap.empty()) return rleData;

    // 初始化第一个位
    int currentBit = (bitmap[0] >> 7) & 1; // 获取第一个字节的最高位
    int count = 0;

    // 遍历位图数据的每一位
    for (const auto& byte : bitmap) {
        for (int i = 7; i >= 0; --i) { // 从每个字节的最高位开始
            int bit = (byte >> i) & 1; // 提取当前位
            if (bit == currentBit) {
                count++; // 当前位与前一个相同，增加计数
            } else {
                // 遇到不同的位，保存当前的游程长度
                rleData.emplace_back(currentBit, count);
                currentBit = bit; // 更新为新的位
                count = 1; // 重置计数
            }
        }
    }

    // 保存最后的游程长度
    rleData.emplace_back(currentBit, count);

    return rleData;
}

double calculateCompressionRatio(size_t originalSize, const std::vector<std::pair<int, int>>& compressedData) {
    
    size_t compressedSize = 0;
    for (const auto& p : compressedData) {
        compressedSize += sizeof(int) + sizeof(int); 
    }

    // 计算压缩比
    return static_cast<double>(originalSize) / compressedSize;
}

// 打印 RLE 压缩数据
void printRLE(const std::vector<std::pair<uint8_t, int>>& rleData) {
    for (const auto& p : rleData) {
        std::cout << "(" << static_cast<int>(p.first) << ", " << p.second << ") ";
    }
    std::cout << std::endl;
}
std::vector<uint8_t> encodeTo3BitBitmap(const std::vector<int>& data) {
    std::vector<uint8_t> bitmap; // 存储结果的位图
    uint8_t currentByte = 0; // 当前字节
    int bitIndex = 0; // 当前位的索引

    for (int value : data) {
        if (value < 0 || value > 6) {
            
            std::cerr << "错误：输入值超出范围 (0-6)。"<<value << std::endl;
            return {};
        }

        // 将当前值的3位插入到当前字节
        currentByte |= (value << (bitIndex)); // 将值左移到正确的位置
        bitIndex += 3; // 增加3位

        // 检查是否需要写入字节
        if (bitIndex >= 8) { // 如果当前字节已填满（8位）
            bitmap.push_back(currentByte); // 将填满的字节加入位图
            bitIndex -= 8; // 减去已填满的位
            currentByte = (value >> (3 - bitIndex)); // 如果有剩余的位，存储到下一个字节的起始位置
        }
    }

    // 如果最后有未完成的字节，写入它
    if (bitIndex > 0) {
        bitmap.push_back(currentByte);
    }

    return bitmap;
}

// 打印位图的二进制表示（每3位一组）
void printBitmap(const std::vector<uint8_t>& bitmap, int totalBits) {
    int bitCount = 0; // 当前已处理的位数
    int cnt = 0;
    for (uint8_t byte : bitmap) {
        if(cnt>10) return;
        for (int i = 7; i >= 0 && bitCount < totalBits; --i, ++bitCount) { // 从最高位开始打印每个字节的二进制位
            std::cout << ((byte >> i) & 1);
            if (bitCount % 3 == 2) { // 每3位打印一个空格
                std::cout << " ";
            }
        }
        cnt++;
    }
    std::cout << std::endl;
}
void cost(std::string filename, std::vector<double> decp_data, std::vector<double> decp_data_copy, std::vector<double> input_data, std::string compressor_id, std::vector<int> delta_counter)
{
    std::vector<int> indexs(input_data.size(), 0);
    std::vector<double> edits;
    std::vector<unsigned long long> exponents;
    std::vector<unsigned long long> mantissas;

    int cnt = 0;
    for (int i=0;i<input_data.size();i++){
        if (decp_data_copy[i]!=decp_data[i]){
            indexs[i] = delta_counter[i];
            if(indexs[i]==6)
            {
                edits.push_back(-(decp_data_copy[i] - (input_data[i] - bound1)));
            }
            cnt++;
        }
    }
    
    // std::string exponentFilename = "exponents.bin";
    // std::string mantissaFilename = "mantissas.bin";

    // // 将数据写入文件
    // writeDataToFile(exponents, exponentFilename);
    // writeDataToFile(mantissas, mantissaFilename);

    // if (compressFile(exponentFilename)) {
    //     std::cout << "Compression of exponents successful." << std::endl;
    // } else {
    //     std::cout << "Compression of exponents failed." << std::endl;
    // }

    // if (compressFile(mantissaFilename)) {
    //     std::cout << "Compression of mantissas successful." << std::endl;
    // } else {
    //     std::cout << "Compression of mantissas failed." << std::endl;
    // }

    // std::uintmax_t original_exponentSize = fs::file_size(exponentFilename);
    // std::uintmax_t compressed_exponentSize = fs::file_size(exponentFilename + ".zst");
    // std::uintmax_t original_mantissaSize = fs::file_size(mantissaFilename);
    // std::uintmax_t compressed_mantissaSize = fs::file_size(mantissaFilename + ".zst");

    // double exponentCompressionRatio = static_cast<double>(original_exponentSize) / compressed_exponentSize;
    // double mantissaCompressionRatio = static_cast<double>(original_mantissaSize) / compressed_mantissaSize;

    // std::cout << "Exponent compression ratio: " << exponentCompressionRatio << std::endl;
    // std::cout << "Mantissa compression ratio: " << mantissaCompressionRatio << std::endl;

    
    std::vector<uint8_t> bitmap = encodeTo3BitBitmap(indexs);

    
    std::string indexfilename = "/pscratch/sd/y/yuxiaoli/MSCz/data1"+filename+".bin";
    std::string editsfilename = "/pscratch/sd/y/yuxiaoli/MSCz/data_edits"+filename+".bin";
    std::string compressedindex = "/pscratch/sd/y/yuxiaoli/MSCz/data1"+filename+".bin.zst";
    std::string compressededits = "/pscratch/sd/y/yuxiaoli/MSCz/data_edits"+filename+".bin.zst";
    
    
    double ratio = double(cnt)/(decp_data_copy.size());
    cout<<cnt<<","<<ratio<<endl;

    std::ofstream file(indexfilename, std::ios::binary | std::ios::out);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(bitmap.data()), bitmap.size());
        file.close();
    } else {
        std::cerr << "cannot open file: " << filename << " ." << std::endl;
    }
    
    std::string command;
    command = "zstd -f " + indexfilename;
    std::cout << "Executing command: " << command << std::endl;
    int result = std::system(command.c_str());
    if (result == 0) {
        
        std::cout << "Compression successful." << std::endl;
    } else {
        std::cout << "Compression failed." << std::endl;
    }

    std::ofstream file1(editsfilename, std::ios::binary | std::ios::out);
    if (file1.is_open()) {
        file1.write(reinterpret_cast<const char*>(edits.data()), edits.size()*sizeof(double));
        file1.close();
    } else {
        std::cerr << "cannot open file: " << filename << " ." << std::endl;
    }
    
    
    command = "zstd -f " + editsfilename;
    std::cout << "Executing command: " << command << std::endl;
    result = std::system(command.c_str());
    if (result == 0) {
        
        std::cout << "Compression successful." << std::endl;
    } else {
        std::cout << "Compression failed." << std::endl;
    }
    
    
    std::uintmax_t compressed_indexSize = fs::file_size(compressedindex);
    std::uintmax_t compressed_editSize =fs::file_size(compressededits);
    std::uintmax_t original_indexSize = fs::file_size(indexfilename);
    std::uintmax_t original_editSize = fs::file_size(editsfilename);
    std::uintmax_t original_dataSize = fs::file_size(inputfilename);
    std::uintmax_t compressed_dataSize = fs::file_size(cpfilename);
    // printf("%f %.f %f %f %f %f\n",static_cast<double>(original_indexSize), static_cast<double>(original_editSize), static_cast<double>(original_dataSize), static_cast<double>(compressed_dataSize), static_cast<double>(compressed_editSize), static_cast<double>(compressed_indexSize));
    double overall_ratio = double(original_dataSize)/(compressed_dataSize+compressed_editSize+compressed_indexSize);
    // cout<<"original_data_size:"<<original_dataSize <<",original_indexSize:"<<compressed_dataSize<<"cr:"<<double(original_dataSize)/compressed_dataSize<<endl;
    // cout<<"original_edit_size:"<<original_editSize <<",original_indexSize:"<<original_indexSize <<"compressed_indexSize:"<<compressed_indexSize<<"compressed_edtsSize:"<<compressed_editSize<<endl;
    double bitRate = 64/overall_ratio; 

    double psnr = calculatePSNR(input_data, decp_data_copy, maxValue-minValue);
    double fixed_psnr = calculatePSNR(input_data, decp_data, maxValue-minValue);

    std::ofstream outFile3("./stat_result/result_"+filename+"_"+compressor_id+"_detailed_additional_time.txt", std::ios::app);

    // 检查文件是否成功打开
    if (!outFile3) {
        std::cerr << "Unable to open file for writing." << std::endl;
        return; // 返回错误码
    }

    
    outFile3 << std::to_string(bound1)<<":" << std::endl;
    outFile3 << std::setprecision(17)<< "related_error: "<<range1 << std::endl;
    outFile3 << "preserve_saddles: "<< preserve_saddles << std::endl;
    outFile3 << "preserve_vpath: "<< preserve_vpath << std::endl;
    outFile3 << "preserve_geometry: "<< preserve_geometry << std::endl;
    outFile3 << "filtration: "<< filtration << std::endl;
    outFile3 << "preserve_connectors: "<< preserve_connectors << std::endl;
    outFile3 << "preserve_types: "<< preserve_type << std::endl;
    outFile3 << "edit_type: "<< edit_type1 << std::endl;
    outFile3 << std::setprecision(17)<< "OCR: "<<overall_ratio << std::endl;
    outFile3 <<std::setprecision(17)<< "CR: "<<double(original_dataSize)/compressed_dataSize << std::endl;
    outFile3 << std::setprecision(17)<<"OBR: "<<bitRate << std::endl;
    outFile3 << std::setprecision(17)<<"BR: "<< 64/(double(original_indexSize)/compressed_dataSize) << std::endl;
    outFile3 << std::setprecision(17)<<"psnr: "<<psnr << std::endl;
    outFile3 << std::setprecision(17)<<"fixed_psnr: "<<fixed_psnr << std::endl;
    

    // outFile3 << std::setprecision(17)<<"right_labeled_ratio: "<<right_labeled_ratio << std::endl;
    outFile3 << std::setprecision(17)<<"edit_ratio: "<<ratio << std::endl;
    outFile3 << std::setprecision(17)<<"compression_time: "<<compression_time<< std::endl;
    outFile3 << std::setprecision(17)<<"additional_time: "<<additional_time<< std::endl;
    outFile3 << "\n" << std::endl;
    // 关闭文件
    outFile3.close();

    std::cout << "Variables have been appended to output.txt" << std::endl;
    return;
}


void original_cost(std::string filename, std::vector<double> decp_data, std::vector<double> decp_data_copy, std::vector<double> input_data, std::string compressor_id)
{
    std::vector<double> indexs;
    std::vector<double> edits;
    int cnt = 0;
    for (int i=0;i<input_data.size();i++){
        
        if (decp_data_copy[i]!=decp_data[i]){
            indexs.push_back(i);
            edits.push_back(decp_data[i]-decp_data_copy[i]);
            cnt++;
        }
    }
    std::vector<int> diffs;  
    std::string indexfilename = "/pscratch/sd/y/yuxiaoli/MSCz/data"+filename+".bin";
    std::string editsfilename = "/pscratch/sd/y/yuxiaoli/MSCz/data_edits"+filename+".bin";
    std::string compressedindex = "/pscratch/sd/y/yuxiaoli/MSCz/data"+filename+".bin.zst";
    std::string compressededits = "/pscratch/sd/y/yuxiaoli/MSCz/data_edits"+filename+".bin.zst";
    if (!indexs.empty()) {
        diffs.push_back(indexs[0]);
    }
    for (size_t i = 1; i < indexs.size(); ++i) {
        diffs.push_back(indexs[i] - indexs[i - 1]);
    }
    double ratio = double(cnt)/(decp_data_copy.size());
    cout<<cnt<<","<<ratio<<endl;
    std::ofstream file(indexfilename, std::ios::binary | std::ios::out);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(diffs.data()), diffs.size() * sizeof(int));
    }
    file.close();
    std::string command;
    command = "zstd -f " + indexfilename;
    std::cout << "Executing command: " << command << std::endl;
    int result = std::system(command.c_str());
    if (result == 0) {
        
        std::cout << "Compression successful." << std::endl;
    } else {
        std::cout << "Compression failed." << std::endl;
    }
    
    std::ofstream file1(editsfilename, std::ios::binary | std::ios::out);
    if (file1.is_open()) {
        file1.write(reinterpret_cast<const char*>(edits.data()), edits.size() * sizeof(double));
    }
    file1.close();

    command = "zstd -f " + editsfilename;
    std::cout << "Executing command: " << command << std::endl;
    result = std::system(command.c_str());
    if (result == 0) {
        std::cout << "Compression successful." << std::endl;
    } else {
        std::cout << "Compression failed." << std::endl;
    }
    // compute result
     std::uintmax_t compressed_indexSize = fs::file_size(compressedindex);
    std::uintmax_t compressed_editSize = fs::file_size(compressededits);
    std::uintmax_t original_indexSize = fs::file_size(indexfilename);
    std::uintmax_t original_editSize = fs::file_size(editsfilename);
    std::uintmax_t original_dataSize = fs::file_size(inputfilename);
    std::uintmax_t compressed_dataSize = fs::file_size(cpfilename);
    // printf("%f %.f %f %f %f %f\n",static_cast<double>(original_indexSize), static_cast<double>(original_editSize), static_cast<double>(original_dataSize), static_cast<double>(compressed_dataSize), static_cast<double>(compressed_editSize), static_cast<double>(compressed_indexSize));
    double overall_ratio = double(original_dataSize)/(compressed_dataSize+compressed_editSize+compressed_indexSize);
    // cout<<"original_data_size:"<<original_dataSize <<",original_indexSize:"<<compressed_dataSize<<"cr:"<<double(original_dataSize)/compressed_dataSize<<endl;
    // cout<<"original_edit_size:"<<original_editSize <<",original_indexSize:"<<original_indexSize <<"compressed_indexSize:"<<compressed_indexSize<<"compressed_edtsSize:"<<compressed_editSize<<endl;
    double bitRate = 64/overall_ratio; 

    double psnr = calculatePSNR(input_data, decp_data_copy, maxValue-minValue);
    double fixed_psnr = calculatePSNR(input_data, decp_data, maxValue-minValue);

    std::ofstream outFile3("./stat_result/result_"+filename+"_"+compressor_id+"_detailed.txt", std::ios::app);

   
    if (!outFile3) {
        std::cerr << "Unable to open file for writing." << std::endl;
        return; 
    }

    
    outFile3 << std::to_string(bound1)<<":" << std::endl;
    outFile3 << std::setprecision(17)<< "related_error: "<<range1 << std::endl;
    outFile3 << "preserve_saddles: "<< preserve_saddles << std::endl;
    outFile3 << "preserve_vpath: "<< preserve_vpath << std::endl;
    outFile3 << "preserve_geometry: "<< preserve_geometry << std::endl;
    outFile3 << "filtration: "<< filtration << std::endl;
    outFile3 << "preserve_connectors: "<< preserve_connectors << std::endl;
    outFile3 << "preserve_types: "<< preserve_type << std::endl;
    outFile3 << "edit_type: "<< edit_type1 << std::endl;
    outFile3 << std::setprecision(17)<< "OCR: "<<overall_ratio << std::endl;
    outFile3 <<std::setprecision(17)<< "CR: "<<double(original_dataSize)/compressed_dataSize << std::endl;
    outFile3 << std::setprecision(17)<<"OBR: "<<bitRate << std::endl;
    outFile3 << std::setprecision(17)<<"BR: "<< 64/(double(original_indexSize)/compressed_dataSize) << std::endl;
    outFile3 << std::setprecision(17)<<"psnr: "<<psnr << std::endl;
    outFile3 << std::setprecision(17)<<"fixed_psnr: "<<fixed_psnr << std::endl;
    

    // outFile3 << std::setprecision(17)<<"right_labeled_ratio: "<<right_labeled_ratio << std::endl;
    outFile3 << std::setprecision(17)<<"edit_ratio: "<<ratio << std::endl;
    outFile3 << std::setprecision(17)<<"compression_time: "<<compression_time<< std::endl;
    outFile3 << std::setprecision(17)<<"additional_time: "<<additional_time<< std::endl;
    outFile3 << "\n" << std::endl;
    // 关闭文件
    outFile3.close();

    std::cout << "Variables have been appended to output.txt" << std::endl;
    return;
}

// first identify the reversed saddle-saddle connector, 
__device__ void getVertexIDsFromEdgeID(
    int edgeID, 
    int width, 
    int height, 
    int depth, 
    
    int &id1, 
    int &id2
    ) {

    // 每个方向的边的数量
    int counts[7] = {
        (width - 1) * height * depth,       // {1, 0, 0} and {-1, 0, 0}
        width * (height - 1) * depth,       // {0, 1, 0} and {0, -1, 0}
        width * height * (depth - 1),       // {0, 0, 1} and {0, 0, -1}
        (width - 1) * (height - 1) * depth, // {-1, 1, 0} and {1, -1, 0}
        (width) * (height - 1) * (depth - 1), // {0, 1, 1} and {0, -1, -1}
        (width - 1) * (height) * (depth - 1), // {-1, 0, 1} and {1, 0, -1}
        (width - 1) * (height - 1) * (depth - 1) // {1, -1, -1} and {-1, 1, 1}
    };

    int accumulated = 0;
    int directionIdx = -1;

    // 找到 edgeID 所属的方向
    for (int i = 0; i < 7; ++i) {
        if (edgeID < accumulated + counts[i]) {
            directionIdx = i;
            break;
        }
        accumulated += counts[i];
    }

    // 如果未找到合法的方向
    if (directionIdx == -1) {
        id1 = id2 = -1;
        return;
    }

    
    // 获取对应的方向
    int dx = directions1[directionIdx * 2][0];
    int dy = directions1[directionIdx * 2][1];
    int dz = directions1[directionIdx * 2][2];

    // 边在该方向上的相对索引
    int edgeInDirection = edgeID - accumulated;
    
    // 根据方向计算基本位置
    int x = 0, y = 0, z = 0;
    int x1, y1, z1;
    if (directionIdx == 0) { // X方向的边
        // 第几层 
        z = edgeInDirection / ((width - 1) * height);
        // 第几行 
        y = (edgeInDirection % ((width - 1) * height)) / (width - 1);
        // 第几个
        x = edgeInDirection % (width - 1);

        x1 = x + 1;
        
        y1 = y;

        z1 = z;
    } else if (directionIdx == 1) { // Y方向的边
        z = edgeInDirection / (width * (height - 1));
        x = (edgeInDirection % (width * (height - 1))) / (height - 1);
        y = edgeInDirection % (height - 1);

        y1 = y + 1;
        x1 = x;
        z1 = z;
    } else if (directionIdx == 2) { // Z方向的边
        y = edgeInDirection / (width * (depth - 1));
        x = (edgeInDirection % (width * (depth - 1))) / (depth - 1);
        z = edgeInDirection % (depth - 1);

        y1 = y;
        x1 = x;
        z1 = z + 1;
    } else if (directionIdx == 3) { // 对角线 {-1, 1, 0} 和 {1, -1, 0}
        int z2 = edgeInDirection / ((width - 1) * (height - 1));
        int y2 = (edgeInDirection % ((width - 1) * (height - 1))) / (width - 1);
        int x2 = edgeInDirection % (width - 1);
        // printf("%d %d %d %d\n", edgeID, x2, y2, z2);
        x = x2 + 1;
        y = y2;
        z = z2;
        y1 = y2 + 1;
        x1 = x2;
        z1 = z2;
    } else if (directionIdx == 4) { // 对角线 {0, 1, 1} 和 {0, -1, -1}
        int x2 = edgeInDirection / ((height - 1) * (depth - 1));
        int y2 = (edgeInDirection % ((height - 1) * (depth - 1))) / (depth - 1);
        int z2 = edgeInDirection % (depth - 1);
        
        // if(edgeID == 15) printf("%d %d %d %d\n", edgeInDirection, x2, y2, z2);
        x = x2;
        y = y2;
        z = z2;
        y1 = y2 + 1;
        x1 = x2;
        z1 = z2 + 1;
        
    } else if (directionIdx == 5) { // 对角线 {-1, 0, 1} 和 {1, 0, -1}
        int y2 = edgeInDirection / ((width - 1) * (depth - 1));
        int z2 = (edgeInDirection % ((width - 1) * (depth - 1))) / (width - 1);
        int x2 = edgeInDirection % (width - 1);


        x = x2 + 1;
        y = y2;
        z = z2;
        y1 = y2;
        x1 = x2;
        z1 = z2 + 1;
    } else if (directionIdx == 6) { // 对角线 {1, -1, -1} 和 {-1, 1, 1}
        int z2 = edgeInDirection /( (width-1) * (height - 1));
        int y2 = (edgeInDirection % ((width - 1) * (height - 1))) / (width - 1);
        int x2 = edgeInDirection % (width - 1);

        x = x2 + 1;
        y = y2;
        z = z2;
        y1 = y2 + 1;
        x1 = x2;
        z1 = z2 + 1;
    }

    // if(edgeID == 99714) printf("%d\n", directionIdx);
    // 计算顶点 ID
    id1 = x + y * width + z * (width * height);
    id2 = x1 + y1 * width + z1 * (width * height);

    // 确保 id1 > id2
    if (id1 < id2) {
        int temp = id1;
        id1 = id2;
        id2 = temp;
    }
}






__global__ void get_vertex(int width, int height, int depth)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_Tetrahedras) return;
   
    int id1, id2, id3, id4;
    int v1 = tetrahedras[index].v1;
    int v2 = tetrahedras[index].v2;
    int v3 = tetrahedras[index].v3;
    int v4 = tetrahedras[index].v4;
    int dir;
    getVerticesFromTetrahedraID(index, width, height, depth, id1, id2, id3, id4);

    if(id1 != v1 || id2 != v2 || id3 != v3 || id4 != v4)
    {
        printf("%d %d %d %d %d %d %d %d %d %d\n", index, num_Edges, id1, v1, id2, v2, id3, v3, id4, v4);
    }
    
    return;

}


__global__ void processVertices(int numVertices, int width, int height, int depth, int type = 0){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numVertices) return;
    // if (change_bool != 0 && changed_index[index] != 1) return ;
   
    double *data1 = input_data;
    int *paired_vertices1 = paired_vertices;
    int *paired_edges1 = paired_edges;
    if(type==1)
    {
        data1 = decp_data;
        paired_vertices1 = dec_paired_vertices;
        paired_edges1 = dec_paired_edges;
    }

    int x = index % width;
    int y = (index / (width)) % height;
    int z = (index / (width * height)) % depth;
    double largest_v = DBL_MAX;
    int largest_id = -1;
    int paired_id = -1;
    bool padding = true;
    int cnt = 0;
    for (int i = 0; i < 14; i++) {
        

        int dx = directions1[i][0];
        int dy = directions1[i][1];
        int dz = directions1[i][2];

        int nx = x + dx;
        int ny = y + dy;
        int nz = z + dz;

        if(nx < 0 || nx >= width || ny < 0 || ny >= height || nz < 0 || nz >= depth) continue;
        int neighbor_id = nx + ny * (width) + nz * (width * height);
        neighbors_index[14*index + cnt] = neighbor_id;
        cnt +=1;
        int edge_id = getEdgeID(x, y, z, nx, ny, nz, width, height, depth);
        
        if ((data1[neighbor_id] < data1[index] or (data1[neighbor_id] == data1[index] and neighbor_id < index)) && (data1[neighbor_id] < largest_v ||(data1[neighbor_id] == largest_v && neighbor_id<largest_id) )) {
            largest_v = data1[neighbor_id];
            largest_id = neighbor_id;
            paired_id = edge_id;
        }
    }

    

    if(paired_id != -1)
    {
        
        paired_vertices1[index] = paired_id;
        paired_edges1[paired_id] = index;
    }
    
    
    return;
}

__global__ void edgePairingKernel(int width, int height, int depth, int numVertices, int numEdges, int numFaces, int type=0, int a2 = 0) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numEdges) return;
    
    
    double *data = input_data;
    int *paired_faces1 = paired_faces;
    int *paired_edges1 = paired_edges;
    if(type==1)
    {
        data = decp_data;
        paired_faces1 = dec_paired_faces;
        paired_edges1 = dec_paired_edges;
        
    }
    
    
    if(a2==1 && paired_edges1[index]!=-1) return;
    if(paired_edges1[index]!=-1) return;
    int v1, v2;
    
    getVertexIDsFromEdgeID(index,width, height, depth , v1, v2);
    // if(blockIdx.x == 194 && threadIdx.x == 386) printf("v1 v2: %d %d\n", v1, v2);
    // if (change_bool != 0 && changed_index[edge->v1->id] != 1 && changed_index[edge->v2->id] != 1) return ;
    int v1_x = v1 % width;
    int v1_y = (v1 / (width)) % height;
    int v1_z = (v1 / (width * height)) % depth;

    int v2_x = v2 % width;
    int v2_y = (v2 / (width)) % height;
    int v2_z = (v2 / (width * height)) % depth;

    double largest_value = DBL_MAX;
    int global_largest_id = INT_MAX;
    int paired_id = -1;

    for(int i=0;i<14;i++)
    {
        
        int nx = v1_x + directions1[i][0];
        int ny = v1_y + directions1[i][1];
        int nz = v1_z + directions1[i][2];
        for(int j=0;j<14;j++)
        {
            int nx1 = v2_x + directions1[j][0];
            int ny1 = v2_y + directions1[j][1];
            int nz1 = v2_z + directions1[j][2];
            int neighbor = nx + ny * width + nz* (height * width);
            
            if(nx == nx1 && ny == ny1 && nz == nz1 && nx >=0 && nx < width && ny >= 0 & ny <height && nz >= 0 && nz<depth && neighbor < num_Vertices && neighbor >=0 )
            {
                
                // edge: v1, v2, e1: v1, v3, e2: v2, v3
                // if(blockIdx.x == 194 && threadIdx.x == 386) printf("neighbor: %d %d\n", neighbor, index);
                int larger_id = v2;
                int larger_v = data[v2];
                if(data[v1]>data[v2] || (data[v1]==data[v2] && v1 > v2))
                {
                    larger_id = v1;
                    larger_v = data[v1];
                }
                // check if could be paired
                int triangleID = getTriangleID(v1, v2, neighbor, width, height, depth );
                // if(blockIdx.x == 194 && threadIdx.x == 386) printf("trianleid: %d %d\n", triangleID, index);
                // printf("%d\n", triangleID);
                // if(neighbor<0 or neighbor >= num_Vertices) printf("%d\n", neighbor);
                if ((data[neighbor] > data[v1] || (data[neighbor] == data[v1] && neighbor > v1)) && a2 == 0) continue;
                if ((data[neighbor] > data[v2] || (data[neighbor] == data[v2] && neighbor > v2)) && a2 == 0) continue;
                
                if((data[neighbor] < largest_value || (data[neighbor]  == largest_value && neighbor < global_largest_id)) && a2 == 0)
                {
                    // if(index==148763 && paired_id!=-1) printf("before: paired here: %d %d %d %d %.17f %.17f %.17f %.17f\n", triangleID, paired_id, neighbor, global_largest_id, data[neighbor], data[global_largest_id], input_data[global_largest_id]-bound, input_data[neighbor]-bound);
                    largest_value = data[neighbor];
                    global_largest_id = neighbor;
                    paired_id = triangleID;
                    // if(index==148763) printf("paired here: %d %d %d %d %.17f %.17f \n", triangleID, paired_id, neighbor, data[neighbor], input_data[neighbor]);
                    // if(paired_id==24824) printf("inside: %d %d %d %d %d\n", paired_id, index, a2, type, paired_faces1[paired_id]);
                    // if(paired_id==24824) printf("inside: %.17f %.17f %.17f %.17f %.17f %.17f\n", decp_data[v1], decp_data[v2], decp_data[neighbor], 
                    // input_data[v1], input_data[v2], input_data[neighbor]);
                    
                }
                
                else if(a2==1)
                {
                    bool c1 = compare_vertices(v1, neighbor, type);
                    bool c2 = compare_vertices(v2, neighbor, type);
                    
                    if(c1!=c2)
                    {
                        
                        if(data[neighbor] < largest_value || (data[neighbor]  == largest_value && neighbor < global_largest_id))
                        {
                            largest_value = data[neighbor];
                            global_largest_id = neighbor;
                            paired_id = triangleID;
                            
                        }
                        // if(index==13256 ) printf("out side: %d %d %d %d %d\n", index, triangleID, a2, type, paired_faces1[triangleID]);
                        // if(index==13256) printf("out side: %d %d %d %d %d %d\n", index, triangleID, v1, v2,neighbor, paired_faces1[triangleID]);
                        // if(index==13256) printf("inside: %.17f %.17f %.17f %.17f %.17f %.17f %d %d \n", decp_data[v1], decp_data[v2], decp_data[neighbor],
                        // input_data[v1], input_data[v2], input_data[neighbor],c1 ,c2);
                    }
                    
                    

                }
            }
        }
    }

    
    
    if(paired_id!=-1 && paired_faces1[paired_id] == -1)
    {
        // if(index==148763) printf("paired here: %d %d\n", index, paired_id);
        paired_edges1[index] = paired_id;
        paired_faces1[paired_id] = index;
        
    }
    return;
    
    
}

__device__ int get_smallest_vertex(int v1, int v2, int v3, int width, int height, int depth)
{
    int v1_x = v1 % width;
    int v1_y = (v1 / (width)) % height;
    int v1_z = (v1 / (width * height)) % depth;

    int v2_x = v2 % width;
    int v2_y = (v2 / (width)) % height;
    int v2_z = (v2 / (width * height)) % depth;

    int v3_x = v3 % width;
    int v3_y = (v3 / (width)) % height;
    int v3_z = (v3 / (width * height)) % depth;

    int global_largest_id = INT_MAX;
    double smallest_value = DBL_MAX;
    int paired_id = -1;
    for (int i = 0; i < 14; i++) 
    {
        int nx = v1_x + directions1[i][0];
        int ny = v1_y + directions1[i][1];
        int nz = v1_z + directions1[i][2];
        int neighbor_id1 = nx + ny * width + nz * width * height;
        
        if(nx >= width || ny >= height || nz >= depth || nx < 0 ||
            ny < 0 || nz < 0 || neighbor_id1 >= num_Vertices || neighbor_id1 < 0 ) continue;

        for (int j = 0; j < 14; j++)
        {
            
            int nx1 = v2_x + directions1[j][0];
            int ny1 = v2_y + directions1[j][1];
            int nz1 = v2_z + directions1[j][2];
            int neighbor_id2 = nx1 + ny1 * width + nz1 * width * height;
            if(nx1 >= width || ny1 >= height || nz1 >= depth || nx1 < 0 ||
                ny1 < 0 || nz1 < 0 || neighbor_id2 >= num_Vertices || neighbor_id2 < 0 ) continue;
            for (int k = 0; k < 14; k++) {

                int nx2 = v3_x + directions1[k][0];
                int ny2 = v3_y + directions1[k][1];
                int nz2 = v3_z + directions1[k][2];
                int neighbor_id3 = nx2 + ny2 * width + nz2 * width * height;
                if(nx2 >= width || ny2 >= height || nz2 >= depth || nx2 < 0 ||
                    ny2 < 0 || nz2 < 0 || neighbor_id3 >= num_Vertices || neighbor_id3 < 0 ) continue;
                
                if(!(neighbor_id1 == neighbor_id2 && neighbor_id2 == neighbor_id3 && neighbor_id1 == neighbor_id3 )) continue;
                if (neighbor_id1 == v1 || neighbor_id1 == v2 || neighbor_id1== v3) continue;

                int v4 = neighbor_id1;
                
                int tetra_id = getTetrahedraID(v1, v2, v3, v4, width, height, depth);
                
                
                
                
                

                bool c1 = compare_vertices(v1, v4, 0);
                bool c2 = compare_vertices(v2, v4, 0);
                bool c3 = compare_vertices(v3, v4, 0);
                
                if((c1 and c2 and !c3) || (c1 and !c2 and c3) || (!c1 and c2 and c3))
                {
                    if((input_data[v4] < smallest_value || (input_data[v4]==smallest_value && v4 < global_largest_id)))
                    {
                        smallest_value = input_data[v4];
                        global_largest_id = v4;
                        paired_id = tetra_id;
                        
                    }
                }
                

                
                }
        }
    }
    if(paired_id!=-1) return global_largest_id;
    return -1;
}


__global__ void trianglePairingKernel(int width, int height, int depth, int numVertices, int numEdges, int numFaces, int numTetrahedra, int type = 0, int a2 = 0) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numFaces) return;
    // printf("%d %d\n", numFaces, num_Tetrahedras);
    
    double *data = input_data;
    int *paired_faces1 = paired_faces;
    int *paired_tetrahedras1 = paired_tetrahedras;
    if (type == 1) {
        data = decp_data;
        paired_faces1 = dec_paired_faces;
        paired_tetrahedras1 = dec_paired_tetrahedras;
    }
    // printf("%d\n", index);
    if(a2==1 && paired_faces1[index]!=-1) return;
    if(paired_faces1[index]!=-1) return;
    int v1, v2, v3;
    
    getVerticesFromTriangleID(index, width, height, depth, v1, v2, v3);
    

    int v1_x = v1 % width;
    int v1_y = (v1 / (width)) % height;
    int v1_z = (v1 / (width * height)) % depth;

    int v2_x = v2 % width;
    int v2_y = (v2 / (width)) % height;
    int v2_z = (v2 / (width * height)) % depth;

    int v3_x = v3 % width;
    int v3_y = (v3 / (width)) % height;
    int v3_z = (v3 / (width * height)) % depth;

    int global_largest_id = INT_MAX;
    double smallest_value = DBL_MAX;
    int paired_id = -1;
    for (int i = 0; i < 14; i++) 
    {
        int nx = v1_x + directions1[i][0];
        int ny = v1_y + directions1[i][1];
        int nz = v1_z + directions1[i][2];
        int neighbor_id1 = nx + ny * width + nz * width * height;
        
        if(nx >= width || ny >= height || nz >= depth || nx < 0 ||
           ny < 0 || nz < 0 || neighbor_id1 >= numVertices || neighbor_id1 < 0 ) continue;

        for (int j = 0; j < 14; j++)
        {
            
            int nx1 = v2_x + directions1[j][0];
            int ny1 = v2_y + directions1[j][1];
            int nz1 = v2_z + directions1[j][2];
            int neighbor_id2 = nx1 + ny1 * width + nz1 * width * height;
            if(nx1 >= width || ny1 >= height || nz1 >= depth || nx1 < 0 ||
                ny1 < 0 || nz1 < 0 || neighbor_id2 >= numVertices || neighbor_id2 < 0 ) continue;
            for (int k = 0; k < 14; k++) {

                int nx2 = v3_x + directions1[k][0];
                int ny2 = v3_y + directions1[k][1];
                int nz2 = v3_z + directions1[k][2];
                int neighbor_id3 = nx2 + ny2 * width + nz2 * width * height;
                if(nx2 >= width || ny2 >= height || nz2 >= depth || nx2 < 0 ||
                    ny2 < 0 || nz2 < 0 || neighbor_id3 >= numVertices || neighbor_id3 < 0 ) continue;
                
                if(!(neighbor_id1 == neighbor_id2 && neighbor_id2 == neighbor_id3 && neighbor_id1 == neighbor_id3 )) continue;
                if (neighbor_id1 == v1 || neighbor_id1 == v2 || neighbor_id1== v3) continue;

                int v4 = neighbor_id1;
                
                int tetra_id = getTetrahedraID(v1, v2, v3, v4, width, height, depth);
                
                if((data[v4] > data[v1] || (data[v4] == data[v1] && v4 > v1)) && a2 == 0) continue;
                if((data[v4] > data[v2] || (data[v4] == data[v2] && v4 > v2)) && a2 == 0) continue;
                if((data[v4] > data[v3] || (data[v4] == data[v3] && v4 > v3)) && a2 == 0) continue;
                
                if((data[v4] < smallest_value || (data[v4]==smallest_value && v4 < global_largest_id)) && a2 == 0)
                {
                    smallest_value = data[v4];
                    global_largest_id = v4;
                    paired_id = tetra_id;
                    
                }
                
                
                else if(a2 == 1)
                {

                    bool c1 = compare_vertices(v1, v4, type);
                    bool c2 = compare_vertices(v2, v4, type);
                    bool c3 = compare_vertices(v3, v4, type);
                    
                    if((c1 and c2 and !c3) || (c1 and !c2 and c3) || (!c1 and c2 and c3))
                    {
                        if((data[v4] < smallest_value || (data[v4]==smallest_value && v4 < global_largest_id)) && a2 == 1)
                        {
                            smallest_value = data[v4];
                            global_largest_id = v4;
                            paired_id = tetra_id;
                            
                        }
                    }
                }

                
                }
        }
    }
    
    
    if(paired_id != -1 && paired_tetrahedras1[paired_id] == -1)
    {
        
        paired_faces1[index] = paired_id;
        paired_tetrahedras1[paired_id] = index;
        // printf("%d\n", paired_id);
    }
    
    
}

__global__ void get_cp_number(int width, int height, int depth, int type=0)
{
    
    
   
    
    int cnt = 0;
    for(int i=0;i<num_Vertices;i++)
    {
        
       
        if(paired_vertices[i]==-1)
        {   
            printf("%d\n", i);
            decp_data[i] = input_data[i] - bound;
            printf("%d\n", i);
            unchangeable_vertices[i] = 1;
            printf("%d\n", i);
            cnt++;
        }
        
    }
    printf("minimum: %d\n", cnt);

    cnt = 0;
    for(int i=0;i<num_Edges;i++)
    {
        
        if(paired_edges[i]==-1) 
        {
            int v1, v2;
            getVertexIDsFromEdgeID(i, width, height, depth, v1, v2);
            decp_data[v1] = input_data[v1] - bound;
            decp_data[v2] = input_data[v2] - bound;
            unchangeable_vertices[v1] = 1;
            unchangeable_vertices[v2] = 1;
            cnt++;
        }
    }
    
    printf("saddle: %d\n", cnt);

    cnt = 0;
    for(int i=0;i<num_Faces;i++)
    {
        
        if(paired_faces[i]==-1)
        {
            int v1, v2, v3;
            getVerticesFromTriangleID(i, width, height, depth, v1, v2, v3);
            decp_data[v1] = input_data[v1] - bound;
            decp_data[v2] = input_data[v2] - bound;
            decp_data[v3] = input_data[v3] - bound;
            unchangeable_vertices[v1] = 1;
            unchangeable_vertices[v2] = 1;
            unchangeable_vertices[v3] = 1;
            
            
            cnt++;
        }
    }


    cnt = 0;
    for(int i=0;i<num_Tetrahedras;i++)
    {
        
        if(paired_tetrahedras[i]==-1)
        {
            int v1, v2, v3, v4;
            getVerticesFromTetrahedraID(i, width, height, depth, v1, v2, v3, v4);
            decp_data[v1] = input_data[v1] - bound;
            decp_data[v2] = input_data[v2] - bound;
            decp_data[v3] = input_data[v3] - bound;
            decp_data[v4] = input_data[v4] - bound;
            unchangeable_vertices[v1] = 1;
            unchangeable_vertices[v2] = 1;
            unchangeable_vertices[v3] = 1;
            unchangeable_vertices[v4] = 1;
            cnt++;
        }
    }
    printf("maximum: %d\n", cnt);

    
}

__global__ void filter_vertices(int width, int height, int depth)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
        
    if(i>=num_Vertices) return;
    
    if(paired_vertices[i]==-1)
    {   
        delta_counter[i] = 6;
        decp_data[i] = input_data[i] - bound;
        unchangeable_vertices[i] = 1;
    }
        
    
}

__global__ void filter_edges(int width, int height, int depth)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
        
    if(i>=num_Edges) return;
    
    if(paired_edges[i]==-1) 
    {
        int v1, v2;
        getVertexIDsFromEdgeID(i, width, height, depth, v1, v2);
        decp_data[v1] = input_data[v1] - bound;
        decp_data[v2] = input_data[v2] - bound;
        unchangeable_vertices[v1] = 1;
        unchangeable_vertices[v2] = 1;
        delta_counter[v1] = 6;
        delta_counter[v2] = 6;
    }
    
}

__global__ void filter_faces(int width, int height, int depth)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
        
    if(i>=num_Faces) return;
    
        
    if(paired_faces[i]==-1)
    {
        int v1, v2, v3;
        getVerticesFromTriangleID(i, width, height, depth, v1, v2, v3);
        decp_data[v1] = input_data[v1] - bound;
        decp_data[v2] = input_data[v2] - bound;
        decp_data[v3] = input_data[v3] - bound;
        unchangeable_vertices[v1] = 1;
        unchangeable_vertices[v2] = 1;
        unchangeable_vertices[v3] = 1;
        delta_counter[v1] = 6;
        delta_counter[v2] = 6;
        delta_counter[v3] = 6;
    }
    
}

__global__ void filter_tetrahedras(int width, int height, int depth)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
        
    if(i>=num_Tetrahedras) return;
    
        
    if(paired_tetrahedras[i]==-1)
    {
        int v1, v2, v3, v4;
        getVerticesFromTetrahedraID(i, width, height, depth, v1, v2, v3, v4);
        decp_data[v1] = input_data[v1] - bound;
        decp_data[v2] = input_data[v2] - bound;
        decp_data[v3] = input_data[v3] - bound;
        decp_data[v4] = input_data[v4] - bound;
        unchangeable_vertices[v1] = 1;
        unchangeable_vertices[v2] = 1;
        unchangeable_vertices[v3] = 1;
        unchangeable_vertices[v4] = 1;
        delta_counter[v1] = 6;
        delta_counter[v2] = 6;
        delta_counter[v3] = 6;
        delta_counter[v4] = 6;
    }
    
}

__global__ void get_false_minimum()
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
        
    if(i>=num_Vertices) return;

    
    if((dec_paired_vertices[i] == -1 && paired_vertices[i] != -1) ||  (dec_paired_vertices[i] != -1 && paired_vertices[i] == -1))
    {
        int idx_fp_min = atomicAdd(&count_f_min, 1);// in one instruction
        
        false_min[idx_fp_min] = i;
    }
}

__global__ void get_false_saddle()
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
        
    if(i>=num_Edges) return;

    
    if((dec_paired_edges[i] == -1 && paired_edges[i] != -1) ||  (dec_paired_edges[i] != -1 && paired_edges[i] == -1))
    {
        int idx_fp_min = atomicAdd(&count_f_saddle, 1);// in one instruction
        
        false_saddle[idx_fp_min] = i;
    }
}

__global__ void get_false_2_saddle()
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
        
    if(i>=num_Faces) return;

    
     if((dec_paired_faces[i] == -1 && paired_faces[i] != -1) ||  (dec_paired_faces[i] != -1 && paired_faces[i] == -1))
    {
        int idx_fp_min = atomicAdd(&count_f_2_saddle, 1);// in one instruction
        
        false_2saddle[idx_fp_min] = i;
    }
}

__global__ void get_false_maximum()
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
        
    if(i>=num_Tetrahedras) return;

    
     if((dec_paired_tetrahedras[i] == -1 && paired_tetrahedras[i] != -1) ||  (dec_paired_tetrahedras[i] != -1 && paired_tetrahedras[i] == -1))
    {
        int idx_fp_min = atomicAdd(&count_f_max, 1);// in one instruction
        
        false_max[idx_fp_min] = i;
    }
}



__global__ void fix_minimum(int width, int height, int depth){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=count_f_min) return;
    int id = false_min[i];
    
    int vertex, v2;
    if(paired_vertices[id]!=-1){
        
        int edge = paired_vertices[id];
        // find the truly
        getVertexIDsFromEdgeID(edge, width, height, depth, vertex, v2);
        if(vertex != id) v2 = vertex;
        // double diff = ((input_data[v2] - bound) - decp_data[v2]) / 2.0 ;
        double diff = get_delta(v2);
        
        double old_value = d_deltaBuffer[v2];
        if (true) {              
            swap(v2, diff);
        } 
        
       
    }
   
    else{
        // double diff = ((input_data[id] - bound) - decp_data[id]) / 2.0 ;
        double diff = get_delta(id);
        double old_value = d_deltaBuffer[id];
        if (true) {              
            swap(id, diff);
        } 
        
    }
}

__global__ void fix_saddle(int width, int height, int depth){
    // get id of the false saddle point -> edge->id;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(i>=count_f_saddle) return;
    int id = false_saddle[i];
    
    int edge = id;
    
    // false positive case: edge is regular in the input data while saddle in the decompressed data.
    if(paired_edges[id]!=-1){
        // printf("1-saddle: %d\n", id);
        // if edge is paired with vertex, one of the edge's vertex is lower than the paired vertex.
        // printf("1-saddle:%d\n", id);
        if(paired_edges[id]<num_Vertices && paired_edges[id]>=0 && paired_edges[id]!=-1 && paired_vertices[paired_edges[id]] == id){
            
            // the paired_vertex's value should be the largest one, so decrease the other one;
            
            int v1, v2;
            
            getVertexIDsFromEdgeID(edge, width, height, depth, v1, v2);
            
            if(v2!=paired_edges[id]) v1 = paired_edges[id];
            else{
                v2 = v1;
                v1 = paired_edges[id];
            }
            
            
            if(dec_paired_vertices[v1] == -1 || dec_paired_vertices[v1] < 0 || dec_paired_vertices[v1] >= num_Edges) return;
            
            int false_edge = dec_paired_vertices[v1];
            
            int false_v1, v2_1;
            getVertexIDsFromEdgeID(false_edge, width, height, depth, false_v1, v2_1);
            int false_v2 = v1 == false_v1?v2_1:false_v1;
            
            
            if(decp_data[v2] > decp_data[false_v2] or (decp_data[v2]==decp_data[false_v2] and v2> false_v2))
            {
                
                // double diff = get_delta(v2);
                double diff = get_delta(v2);
        
                double old_value = d_deltaBuffer[v2];
                if (true) {              
                    swap(v2, diff);
                }
                // if(diff>1e-16) dec_vertices[v2->id].value = ((input_data[v2->id] - bound) + dec_vertices[v2->id].value)/2.0;
                // else dec_vertices[v2->id].value = (input_data[v2->id] - bound);
                
                
            }
        }
        // if edge is paired with face in the original data, then the edge's value should be the highest.
        else if(paired_edges[id]!=-1 && paired_edges[id]<num_Faces &&paired_edges[id]>=0 && paired_faces[paired_edges[id]] == id){
                
                
                // if( blockIdx.x == 0) printf("2: %d\n", i);
                int face = paired_edges[id];
                
                int v1, v2;
                getVertexIDsFromEdgeID(edge, width, height, depth, v1, v2);
                // if( blockIdx.x == 0) printf("v1 v2: %d %d\n", v1, v2);
                int v1_1, v2_1, v3;
                getVerticesFromTriangleID(face, width, height, depth, v1_1, v2_1, v3);
                // if( blockIdx.x == 0) printf("v1 v2 v3: %d %d %d\n", v1_1, v2_1, v3);
                
                
                if(v1_1!=v1 and v1_1!=v2) v3 = v1_1;
                else if(v2_1!=v1 and v2_1!=v2) v3 = v2_1;
               
                int sortedV1 = compare_vertices(v1, v2, 1)? v1:v2;
                int sortedV2 = compare_vertices(v1, v2, 1)? v2:v1;


                int other_sortedV1 = !compare_vertices(v1, v3, 1)? v3:v1;
                int other_sortedV2 = !compare_vertices(v1, v3, 1)? v1:v3;
                
                
                

                int other_sortedV1_1 = !compare_vertices(v2, v3, 1)? v3:v2;
                int other_sortedV2_1 = !compare_vertices(v2, v3, 1)? v2:v3;
                
                
                
                // if edge is paired with face, then edge(v1,v2) > edge(v2,v3) and edge(v1, v3);
                bool v1_edge = compare_vertices(v1, v3, 1);
                bool v2_edge = compare_vertices(v2, v3, 1);

                
                bool o_v1_edge = compare_vertices(v1, v3, 0);
                bool o_v2_edge = compare_vertices(v2, v3, 0);
                
                
                if(v1_edge and v2_edge){
                    
                     
                    printf("saddle-1: %d %d %d %d %d\n", id, dec_paired_faces[face], dec_paired_edges[id],paired_edges[id], dec_paired_edges[dec_paired_faces[face]]);
                    
                    
                   
                        // double diff = get_delta(v3);
                        double diff = get_delta(v3);
                        double old_value = d_deltaBuffer[v3];
                        if (true) {              
                            swap(v3, diff);
                        }
                    
               
                }
                else if(o_v1_edge and o_v2_edge)
                {
                    // double diff = get_delta(v3);
                    double diff = get_delta(v3);
                    double old_value = d_deltaBuffer[v3];
                    if (true) {              
                        swap(v3, diff);
                    }
                }


                // paired edge with al 2.
                else
                {
                    // edge in the original data is paired in the second round
                    
                    // if is paired with a face, find the face
                    // find the faces paired cell in the decompress data to see if it is paired with any other cell in the first round;
                    // if is edge;
                    if(paired_edges[id] != -1 && paired_edges[id] < num_Faces && paired_edges[id] >=0 && dec_paired_faces[paired_edges[id]] != -1 && dec_paired_edges[dec_paired_faces[paired_edges[id]]] == paired_edges[id] )
                    {
                    //    printf("2-saddle: %d %d %d %d %d\n",id, o_v1_edge, o_v2_edge, v1_edge, v2_edge);
                        
                        // if is edge
                        int false_id = dec_paired_faces[paired_edges[id]];
                        
                        int false_edge = false_id;
                        edge = false_edge;
                        int v1_1, v2_1;
                        getVertexIDsFromEdgeID(false_edge, width, height, depth, v1_1, v2_1);
                        int false_v = (v1_1 != v1 && v1_1 != v2)?v1_1:v2_1;
                        
                        
                        // double diff = (input_data[false_v] - bound - decp_data[false_v])/2.0;
                        double diff = get_delta(false_v);
                        // printf("%d %.17f %d %d\n", false_v->id, diff, edges[false_edge.id].paired_face_id, false_edge.paired_face_id);
                        
                        int v1, v2;
                        getVertexIDsFromEdgeID(edge, width, height, depth, v1, v2);
                        int v1_2, v2_2, v3;
                        getVerticesFromTriangleID(face, width, height, depth, v1_2, v2_2, v3);
                        
                        
                        if(v1_2!=v1 and v1_2!=v2) v3 = v1_2;
                        else if(v2_2!=v1 and v2_2!=v2) v3 = v2_2;
                        
                        sortedV1 = !compare_vertices(v1, v2, 1)? v2:v1;
                        sortedV2 = !compare_vertices(v1, v2, 1)? v1:v2;


                        other_sortedV1 = !compare_vertices(v1, v3, 1)? v3:v1;
                        other_sortedV2 = !compare_vertices(v1, v3, 1)? v1:v3;
                        
                        
                        

                        other_sortedV1_1 = !compare_vertices(v2, v3, 1)? v3:v2;
                        other_sortedV2_1 = !compare_vertices(v2, v3, 1)? v2:v3;
                        
                        
                        
                        // if edge is paired with face, then edge(v1,v2) > edge(v2,v3) and edge(v1, v3);
                        bool v1_edge = compare_vertices(v2, v3, 1);
                        bool v2_edge = compare_vertices(v1, v3, 1);
                        
                        o_v1_edge = compare_vertices(v2, v3, 0);
                        o_v2_edge = compare_vertices(v1, v3, 0);
                        
                        if(o_v1_edge and o_v2_edge and paired_edges[false_edge]!=-1 && paired_faces[paired_edges[false_edge]] == false_edge)
                        {
                            // printf("2-saddle: %d %d %d %d %d\n",id, o_v1_edge, o_v2_edge, v1_edge, v2_edge);
                            // if( blockIdx.x == 0) printf("2: v1 v2 v3: %d %d %d %d %d %d %d\n", v1, v2, v1_2, v2_2, v3, threadIdx.x, paired_edges[false_edge]);
                            int f_id = paired_edges[false_edge];
                            int F = f_id;
                            int v1_2, v2_2, v3;
                            getVerticesFromTriangleID(f_id, width, height, depth, v1_2, v2_2, v3);
                            if(v1_2!=v1 and v1_2!=v2) v3 = v1_2;
                            else if(v2_2!=v1 and v2_2!=v2) v3 = v2_2;
                            // diff = get_delta(v3);
                            
                            diff = get_delta(v3);
                            double old_value = d_deltaBuffer[v3];
                            if (true) {              
                                swap(v3, diff);
                            }
                        }
                        
                        else if(abs(diff)>0)
                        {
                            // diff = (input_data[false_v] - bound - decp_data[false_v])/2.0;
                            diff = get_delta(false_v);
                            double old_value = d_deltaBuffer[false_v];
                            if (true) {              
                                swap(false_v, diff);
                            }
                        }
                        
                        
                        
                    }
                    
                    else{
                        if(o_v1_edge and !v1_edge)
                        {
                            // if( blockIdx.x == 0) printf("2: v1 v2 v3: %d %d %d %d\n", v1_1, v2_1, v3);
                            // double diff = (input_data[other_sortedV1] - bound - decp_data[other_sortedV1])/2.0;
                            double diff = get_delta(other_sortedV1);
                            double old_value = d_deltaBuffer[other_sortedV1];
                            if (true) {              
                                swap(other_sortedV1, diff);
                            }
                        }

                        else
                        {
                            // if( blockIdx.x == 0) printf("3: v1 v2 v3: %d %d %d %d\n", v1_1, v2_1, v3);
                            // double diff = (input_data[other_sortedV1_1] - bound - decp_data[other_sortedV1_1])/2.0;
                            double diff = get_delta(other_sortedV1_1);
                            double old_value = d_deltaBuffer[other_sortedV1_1];
                            if (true) {              
                                swap(other_sortedV1_1, diff);
                            }
                        }
                    }
                    
                }
                
            }
    }
    // false negative case: edge is saddle in the input data while regular in the decompressed data.
    else{
        //  if( blockIdx.x == 0) printf("1: %d %d\n", id, threadIdx.x);
        // if( blockIdx.x == 0) printf("2: %d\n", i);
        // if edge is paired with vertex in the decompressed data/
        // decrease the value of the paried vertex;
        // std::cout<<"yes3"<<std::endl;
        // if(id==33000) cout<<"2"<<endl;
        // printf("fn: %d\n", id);
        
        if(dec_paired_edges[id]!=-1 && dec_paired_edges[id]<num_Vertices && dec_paired_edges[id]>=0 && dec_paired_vertices[dec_paired_edges[id]] == id)
        {
            // printf("saddle-2: %d\n", id);
            int v_id = dec_paired_edges[id];
            
            double diff = get_delta(v_id);
            double old_value = d_deltaBuffer[v_id];
            if (true) {              
                swap(v_id, diff);
            }
            
        }
        // if edge is paired with a face in the decompressed data/
        else if(dec_paired_edges[id]!=-1 && dec_paired_edges[id]<num_Faces && dec_paired_edges[id]>=0 && dec_paired_faces[dec_paired_edges[id]] == id) {
            
            int face = dec_paired_edges[id];
            int v1, v2;
            getVertexIDsFromEdgeID(edge, width, height, depth,  v1, v2);
            int v1_1, v2_1, v3;
            getVerticesFromTriangleID(face, width, height, depth, v1_1, v2_1, v3);
            

            if(v1_1!=v1 and v1_1!=v2) v3 = v1_1;
            else if(v2_1!=v1 and v2_1!=v2) v3 = v2_1;
            

            

            int sortedV1 = !compare_vertices(v1, v2, 1)? v2:v1;
            int sortedV2 = !compare_vertices(v1, v2, 1)? v1:v2;


            int other_sortedV1 = !compare_vertices(v1, v3, 1)? v3:v1;
            int other_sortedV2 = !compare_vertices(v1, v3, 1)? v1:v3;
            
            
            // Edge e1(v1, &v3, 0);

            int other_sortedV1_1 = !compare_vertices(v2, v3, 1)? v3:v2;
            int other_sortedV2_1 = !compare_vertices(v2, v3, 1)? v2:v3;
            
            // Edge e2(v2, &v3, 0);
            
            

            
            // if edge is paired with face, then edge(v1,v2) > edge(v2,v3) and edge(v1, v3);
            bool v1_edge = compare_vertices(v2, v3, 1);
            bool v2_edge = compare_vertices(v1, v3, 1);

            bool or_v1_edge = compare_vertices(v2, v3, 0);
            bool or_v2_edge =  compare_vertices(v1, v3, 0);
            if(v1_edge == v2_edge)
            {
                // printf("1-saddle: %d\n", id);
                if(v1_edge!=or_v1_edge)
                {
                
                    double diff = get_delta(v2);
                    double old_value = d_deltaBuffer[v2];
                    if (true) {              
                        swap(v2, diff);
                    }
                }
                if(v2_edge!=or_v2_edge){

                    // double diff = 
                    double diff = get_delta(v1);
                    double old_value = d_deltaBuffer[v1];
                    if (true) {              
                        swap(v1, diff);
                    }
                    
                    
                }
            }
            
            else
            {
                
                if(paired_edges[id]!=-1)
                {
                    
                    int t = paired_edges[id];
                    if(t>=0 && t<num_Vertices && paired_vertices[t] == id)
                    {
                        int m = v1;
                        if(v1==t) m = v2;
                        double diff = get_delta(m);
                        double old_value = d_deltaBuffer[m];
                        if (true) {              
                            swap(m, diff);
                        }
                    }

                    else
                    {

                    }
                }
                
                else
                {
                    int v1_x = v1 % width;
                    int v1_y = (v1 / (width)) % height;
                    int v1_z = (v1 / (width * height)) % depth;

                    int v2_x = v2 % width;
                    int v2_y = (v2 / (width)) % height;
                    int v2_z = (v2 / (width * height)) % depth;
                    double largest_value = DBL_MAX;
                    int global_largest_id = INT_MAX;
                    int paired_id = -1;
                    for(int i=0;i<14;i++)
                    {
                        
                        int nx = v1_x + directions1[i][0];
                        int ny = v1_y + directions1[i][1];
                        int nz = v1_z + directions1[i][2];
                        for(int j=0;j<14;j++)
                        {
                            int nx1 = v2_x + directions1[j][0];
                            int ny1 = v2_y + directions1[j][1];
                            int nz1 = v2_z + directions1[j][2];
                            int neighbor = nx + ny * width + nz* (height * width);
                            
                            if(nx == nx1 && ny == ny1 && nz == nz1 && nx >=0 && nx < width && ny >= 0 & ny <height && nz >= 0 && nz<depth && neighbor < num_Vertices && neighbor >=0 )
                            {
                                
                                // edge: v1, v2, e1: v1, v3, e2: v2, v3
                                // if(blockIdx.x == 194 && threadIdx.x == 386) printf("neighbor: %d %d\n", neighbor, index);
                                int larger_id = v2;
                                int larger_v = input_data[v2];
                                if(input_data[v1]>input_data[v2] || (input_data[v1]==input_data[v2] && v1 > v2))
                                {
                                    larger_id = v1;
                                    larger_v = input_data[v1];
                                }
                                // check if could be paired
                                int triangleID = getTriangleID(v1, v2, neighbor, width, height, depth );
                                // if(blockIdx.x == 194 && threadIdx.x == 386) printf("trianleid: %d %d\n", triangleID, index);
                                // printf("%d\n", triangleID);
                                // if(neighbor<0 or neighbor >= num_Vertices) printf("%d\n", neighbor);
                                
                                    bool c1 = compare_vertices(v1, neighbor, 0);
                                    bool c2 = compare_vertices(v2, neighbor, 0);
                                    
                                    if(c1!=c2)
                                    {
                                        
                                        if(input_data[neighbor] < largest_value || (input_data[neighbor]  == largest_value && neighbor < global_largest_id))
                                        {
                                            largest_value = input_data[neighbor];
                                            global_largest_id = neighbor;
                                            paired_id = triangleID;
                                            
                                        }
                                    }
                                    
                                    

                                
                            }
                        }
                    }
                    if(paired_id!=-1)
                    {
                        int m = global_largest_id;
                        double diff = get_delta(m);
                        double old_value = d_deltaBuffer[m];
                        if (true) {              
                            swap(m, diff);
                        }
                        if(id == 13256) printf("%d %d\n", m, v3);
                        if(id == 13256) printf("saddle-2: %d %d %d %d %d %d %d %d\n", id, face, paired_faces[face], dec_paired_edges[id], v1_edge, v2_edge, or_v1_edge, or_v2_edge);
                        if(id == 13256) printf("saddle-2: %d %.17f %.17f %.17f %.17f %.17f %.17f\n", id, decp_data[v1], decp_data[v2], decp_data[v3], input_data[v1] - bound, input_data[v2] -bound, input_data[v3] - bound);
                    }
                    
                    else
                    {
                        
                        for(int m:{v1,v2})
                        {
                            double diff = get_delta(m);
                            double old_value = d_deltaBuffer[m];
                            if (true) {              
                                swap(m, diff);
                            }
                        }
                    }

                   
                    
                }
                        
                    }
                }
    }
}

__global__ void fix_2saddle(int width, int height, int depth){
    // get id of the false saddle point -> edge->id;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(i>=count_f_2_saddle) return;
    // printf("%d %d\n", i, threadIdx.x);
    int id = false_2saddle[i];
    
    
   
    
    int face = id;
    
    // false positive case: edge is regular in the input data while saddle in the decompressed data.
    if(paired_faces[id]!=-1){
        
        // if edge is paired with vertex, one of the edge's vertex is lower than the paired vertex.
        
        if(paired_faces[id]<num_Edges && paired_faces[id]>=0 && paired_edges[paired_faces[id]] == id){
            // printf("2-saddle-1: %d\n", id);
            // the paired_vertex's value should be the largest one, so decrease the other one;
            int edge = paired_faces[id];
            int v1, v2;
            getVertexIDsFromEdgeID(edge, width, height, depth, v1, v2);
            // if(threadIdx.x == 38 && blockIdx.x == 62) printf("%d %d\n", v1, v2);
            int v1_1, v2_1, v3;
            getVerticesFromTriangleID(face, width, height, depth, v1_1, v2_1, v3);
            // if(threadIdx.x == 38 && blockIdx.x == 62) printf("%d %d %d\n", v1_1, v2_1,v3);
            
            if(v1_1!=v1 and v1_1!=v2) v3 = v1_1;
            else if(v2_1!=v1 and v2_1!=v2) v3 = v2_1;
            
    
            int sortedV1 = !compare_vertices(v1, v2, 1)? v2:v1;
            int sortedV2 = !compare_vertices(v1, v2, 1)? v1:v2;


            int other_sortedV1 = !compare_vertices(v1, v3, 1)? v3:v1;
            int other_sortedV2 = !compare_vertices(v1, v3, 1)? v1:v3;
            
            
            // Edge e1(v1, &v3, 0);

            int other_sortedV1_1 = !compare_vertices(v2, v3, 1)? v3:v2;
            int other_sortedV2_1 = !compare_vertices(v2, v3, 1)? v2:v3;
            
            // Edge e2(v2, &v3, 0);
            
            
            // if edge is paired with face, then edge(v1,v2) > edge(v2,v3) and edge(v1, v3);
            bool v1_edge = compare_vertices(v2, v3, 1);
            bool v2_edge = compare_vertices(v1, v3, 1);
            
            
            bool o_v1_edge = compare_vertices(v2, v3, 0);
            bool o_v2_edge = compare_vertices(v1, v3, 0);

            // if(count_f_2_saddle == 10 && id == 1613367) printf("%d %d %d %d %d\n",id, v1_edge, v2_edge, o_v1_edge, o_v2_edge);
            if(v1_edge and v2_edge)
            {
                
                int f_id = paired_edges[edge];
                if(f_id == -1 || paired_faces[f_id] != edge) return;
                face = f_id;
                
                int v1_1, v2_1, v3;
                getVerticesFromTriangleID(face, width, height, depth, v1_1, v2_1, v3);
                if(v1_1!=v1 and v1_1!=v2) v3 = v1_1;
                else if(v2_1!=v1 and v2_1!=v2) v3 = v2_1;
                double diff = get_delta(v3);
                
        
                double old_value = d_deltaBuffer[v3];
                if (true) {              
                    swap(v3, diff);
                }

            }

            

            else if(o_v1_edge != o_v2_edge and v1_edge != v2_edge)
            {
                
                int f_id = dec_paired_edges[edge];
                if(f_id == -1 || (f_id != -1 && dec_paired_faces[dec_paired_edges[edge]] != edge ))
                {
                    double diff = get_delta(v3);
                
        
                    double old_value = d_deltaBuffer[v3];
                    if (true) {              
                        swap(v3, diff);
                    }
                }

                else if(dec_paired_edges[edge]<num_Faces && dec_paired_edges[edge]>=0)
                {
                    int F = f_id;

                    int v1_2, v2_2, v3_1;
                    getVerticesFromTriangleID(F, width, height, depth, v1_2, v2_2, v3_1);
                    if(v1_2!=v1 and v1_2!=v2) v3_1= v1_2;
                    else if(v2_2!=v1 and v2_2!=v2) v3_1 = v2_2;
                    


                    v1_edge = compare_vertices(v2, v3_1, 1);
                    v2_edge = compare_vertices(v1, v3_1, 1);

                    

                    o_v1_edge = compare_vertices(v2, v3_1, 0);
                    o_v2_edge = compare_vertices(v1, v3_1, 0);

                    if(v1_edge and v2_edge and ((!o_v1_edge and o_v2_edge) or (o_v1_edge and !o_v2_edge)) )
                    {
                    if(!o_v1_edge)
                    {
                        
                        int sortedV1 = !compare_vertices(v1, v2, 1)? v2:v1;
                        // int sortedV2 = compare_vertices(e3->v1, e3->v2)? e3->v1:e3->v2;
                        double diff = get_delta(v2);
                        // printf("%.17f %.17f %.17f %.17f %.17f %.17f\n", decp_data[v2], input_data[v2], decp_data[v3_1.id], input_data[v3_1.id], decp_data[v1], input_data[v1]);

                        double old_value = d_deltaBuffer[v2];
                        if (true) {              
                            swap(v2, diff);
                        }
                    }

                    else{
                        double diff = get_delta(v1);
                
        
                        double old_value = d_deltaBuffer[v1];
                        if (true) {              
                            swap(v1, diff);
                        }
                    }
                }

                else
                {
                    
                    
                        double diff = get_delta(v3);
                
        
                        double old_value = d_deltaBuffer[v3];
                        if (true) {              
                            swap(v3, diff);
                        }
                    }
                    
                // }

                }
                

                
                
                
                
            }

            else if(!compare_vertices(v1, v3, 1) or !compare_vertices(v2, v3, 1))
            {
                
                double diff = get_delta(v3);
                
        
                double old_value = d_deltaBuffer[v3];
                if (true) {              
                    swap(v3, diff);
                }
                
                
            }

            



        }
        // if face is paired with face in the original data, then the faces's value should be the highest.
        
        else if(paired_faces[id] != -1&& paired_faces[id] < num_Tetrahedras &&paired_faces[id]>=0 && paired_tetrahedras[paired_faces[id]] == id )
        {
                //  if(threadIdx.x == 38 && blockIdx.x == 62) printf("2%d\n", i);
                
                int tetrahedra = paired_faces[id];
                
                int v1, v2, v3;
                getVerticesFromTriangleID(id, width, height, depth, v1, v2, v3);
                int v1_1, v2_1, v3_1, v4;
                getVerticesFromTetrahedraID(tetrahedra , width, height, depth, v1_1, v2_1, v3_1, v4);

                if(v1_1!=v1 and v1_1!=v2 and v1_1 != v3) v4 = v1_1;
                else if(v2_1!=v1 and v2_1!=v2 and v2_1 != v3) v4 = v2_1;
                else if(v3_1!=v1 and v3_1!=v2 and v3_1 != v3) v4 = v3_1;

                // Face f1(v1,v2,&v4);
                // Face f2(v1,v3,&v4);
                // Face f3(v2,v3,&v4);
                
                int sortedV1 = v1, sortedV2 = v2, sortedV3 = v3;
                int sortedV1_a = v1, sortedV2_a = v2, sortedV3_a = v4;
                int sortedV1_b = v1, sortedV2_b = v3, sortedV3_b = v4;
                int sortedV1_c = v2, sortedV2_c = v3, sortedV3_c = v4;
            
                // if edge is paired with face, then edge(v1,v2) > edge(v2,v3) and edge(v1, v3);
                bool f1_face = compare_vertices(v3, v4, 1);
                bool f2_face = compare_vertices(v2, v4, 1);
                bool f3_face = compare_vertices(v1, v4, 1);
                
                
                bool or_f1_face = compare_vertices(v3, v4, 0);
                bool or_f2_face = compare_vertices(v2, v4, 0);
                bool or_f3_face = compare_vertices(v1, v4, 0);
                
                if(!or_f1_face || !or_f2_face || !or_f3_face)
                {
                    
                    
                    // int m = get_smallest_vertex(v1, v2, v3, width, height, depth);
                    // if(m!=-1) v4 = m;
                    
                    int paired_false_face = dec_paired_tetrahedras[tetrahedra];
                    // if(id == 375634) printf("2-saddle-2: %d %d %d %d %d %d %d %d %d\n", id, or_f1_face, or_f2_face, or_f3_face, f1_face, f2_face, f3_face, paired_false_face);
                    if(paired_false_face!=-1)
                    {
                        // printf("2-saddle: %d\n", paired_tetrahedras[tetrahedra]);
                        // paired_false_face = paired_tetrahedras[tetrahedra];
                        int v1_1, v2_1, v3_1;
                        getVerticesFromTriangleID(paired_false_face, width, height, depth, v1_1, v2_1, v3_1);
                        bool f1_face = compare_vertices(v3_1, v4, 1);
                        bool f2_face = compare_vertices(v2_1, v4, 1);
                        bool f3_face = compare_vertices(v1_1, v4, 1);
                        
                        
                        bool or_f1_face = compare_vertices(v3_1, v4, 0);
                        bool or_f2_face = compare_vertices(v2_1, v4, 0);
                        bool or_f3_face = compare_vertices(v1_1, v4, 0);

                        if((!f1_face || !f2_face || !f3_face) && (!or_f1_face || !or_f2_face || !or_f3_face))
                        {
                            int id = paired_faces[paired_false_face];
                            //  
                            if(id>=0 && id < num_Edges && paired_edges[id] == paired_false_face)
                            {
                                int v1, v2;
                                getVertexIDsFromEdgeID(id, width, height, depth, v1, v2);
                                int v3 = v3_1;
                                if(v1_1!=v1 && v2!=v1_1)  v3 = v1_1;
                                else if(v2_1!=v1 && v2_1 != v2) v3 = v2_1;
                                double diff = get_delta(v3);
                                double old_value = d_deltaBuffer[v3];
                                if (true) {              
                                    swap(v3, diff);
                                }

                            }
                            else if(id>=0 && id < num_Tetrahedras && paired_tetrahedras[id] == paired_false_face)
                            {
                                // printf("2-saddle tetra: %d %d %d %d %d %d %d\n", f1_face, f2_face, f3_face, or_f1_face, or_f2_face, or_f3_face, id);
                                int v1,v2,v3,v4;
                                getVerticesFromTetrahedraID(id, width, height, depth, v1,v2,v3,v4);
                                if(v1!=v1_1 && v1!=v2_1 && v1!=v3_1) v4 = v1_1;
                                else if(v2!=v1_1 && v2!=v2_1 && v2!=v3_1) v4 = v2_1;
                                else if(v3!=v1_1 && v3!=v2_1 && v3!=v3_1) v4 = v3_1;

                                double diff = get_delta(v4);
                                double old_value = d_deltaBuffer[v4];
                                for(int m:{v1,v2,v3,v4})
                                {
                                    double diff = get_delta(m);
                                    double old_value = d_deltaBuffer[m];
                                    if (true) {              
                                        swap(m, diff);
                                    }
                                }
                            }
                        }
                        // if(id == 375634)
                        else
                        {
                            for(int m:{v1_1, v2_1, v3_1})
                            {
                                double diff = get_delta(m);
                                double old_value = d_deltaBuffer[m];
                                if (true) {              
                                    swap(m, diff);
                                }
                            }
                        }
                        
                    }
                    
                    else
                    {
                        double diff = get_delta(v4);
                        double old_value = d_deltaBuffer[v4];
                        if (true) {              
                            swap(v4, diff);
                        }
                    }

                    
                    
                    
                }
                else 
                {
                    double diff = get_delta(v4);
                    double old_value = d_deltaBuffer[v4];
                    if (true) {              
                        swap(v4, diff);
                    }
                    // if(!f1_face){
                    
                    
                    //     // if(!compare_vertices(sortedV1, sortedV1_a, 1))
                    //     // {
                    //     //     double diff = (input_data[v3] - bound - decp_data[sortedV1_a])/2.0;
                    //     //     double old_value = d_deltaBuffer[sortedV1_a];
                    //     //     if (true) {              
                    //     //         swap(sortedV1_a, diff);
                    //     //     }
                            
                    //     // }

                    //     // else if(!compare_vertices(sortedV2, sortedV2_a, 1)) 
                    //     // {
                    //     //         double diff = get_delta(sortedV2_a);
                    //     //         double old_value = d_deltaBuffer[sortedV2_a];
                    //     //         if (true) {              
                    //     //             swap(sortedV2_a, diff);
                    //     //         }
                            
                    //     // }

                    //     // else if(!compare_vertices(sortedV3, sortedV3_a, 1)) 
                    //     // {
                    //     //         double diff = (input_data[sortedV3_a] - bound - decp_data[sortedV3_a])/2.0;
                    //     //         double old_value = d_deltaBuffer[sortedV3_a];
                    //     //         if (true) {              
                    //     //             swap(sortedV3_a, diff);
                    //     //         }
                            
                    //     // }
                    //     printf("2-saddle-2: %d\n", id);
                    //     double diff = get_delta(v4);
                    //     double old_value = d_deltaBuffer[v4];
                    //     if (true) {              
                    //         swap(v4, diff);
                    //     }
                        
                    // }

                    // if(!f2_face){
                        
                    //     double diff = get_delta(v4);
                    //     double old_value = d_deltaBuffer[v4];
                    //     if (true) {              
                    //         swap(v4, diff);
                    //     }
                    //     // if(!compare_vertices(sortedV1, sortedV1_b, 1))
                    //     // {
                    //     //     double diff = get_delta(sortedV1_b);
                    //     //     double old_value = d_deltaBuffer[sortedV1_b];
                    //     //     if (true) {              
                    //     //         swap(sortedV1_b, diff);
                    //     //     }
                            
                    //     // }

                    //     // else if(!compare_vertices(sortedV2, sortedV2_b, 1)) 
                    //     // {
                    //     //         double diff = get_delta(sortedV2_b);
                    //     //         double old_value = d_deltaBuffer[sortedV2_b];
                    //     //         if (true) {              
                    //     //             swap(sortedV2_b, diff);
                    //     //         }
                            
                    //     // }

                    //     // else if(!compare_vertices(sortedV3, sortedV3_b, 1)) 
                    //     // {
                    //     //         double diff = (input_data[sortedV3_b] - bound - decp_data[sortedV3_b])/2.0;
                    //     //         double old_value = d_deltaBuffer[sortedV3_b];
                    //     //         if (true) {              
                    //     //             swap(sortedV3_b, diff);
                    //     //         }
                            
                    //     // }
                        
                    // }

                    // if(!f3_face){
                    //     double diff = get_delta(v4);
                    //     double old_value = d_deltaBuffer[v4];
                    //     if (true) {              
                    //         swap(v4, diff);
                    //     }
                    //     // if(!compare_vertices(sortedV1, sortedV1_c, 1))
                    //     // {
                    //     //     double diff = get_delta(sortedV1_c);
                    //     //     double old_value = d_deltaBuffer[sortedV1_c];
                    //     //     if (true) {              
                    //     //         swap(sortedV1_c, diff);
                    //     //     }
                            
                    //     // }

                    //     // else if(!compare_vertices(sortedV2, sortedV2_c, 1)) 
                    //     // {
                    //     //         double diff = get_delta(sortedV2_c);
                    //     //         double old_value = d_deltaBuffer[sortedV2_c];
                    //     //         if (true) {              
                    //     //             swap(sortedV2_c, diff);
                    //     //         }
                            
                    //     // }

                    //     // else if(!compare_vertices(sortedV3, sortedV3_c, 1)) 
                    //     // {
                    //     //         double diff = (input_data[sortedV3_c] - bound - decp_data[sortedV3_c])/2.0;
                    //     //         double old_value = d_deltaBuffer[sortedV3_c];
                    //     //         if (true) {              
                    //     //             swap(sortedV3_c, diff);
                    //     //         }
                            
                    //     // }
                    // }
                }
                

        }
    }
    // false negative case: edge is saddle in the input data while regular in the decompressed data.
    else{
        
        // printf("%d %d\n", id, threadIdx.x);
        // if edge is paired with vertex in the decompressed data/
        // decrease the value of the paried vertex;
        // std::cout<<"yes3"<<std::endl;
        // if(id==33000) cout<<"2"<<endl;
        
        if(dec_paired_faces[id] != -1 && dec_paired_faces[id]<num_Edges && dec_paired_edges[dec_paired_faces[id]] == id )
        {
            // printf("2-saddle-3: %d\n", id);
            int edge_id = dec_paired_faces[id];
            // check the edge's paired face, find the v3, decrease it.
            int v1, v2;
            getVertexIDsFromEdgeID(edge_id, width, height, depth, v1, v2);
            int sortedV1 = !compare_vertices(v1, v2, 1)? v2:v1;
            int sortedV2 = !compare_vertices(v1, v2, 1)? v1:v2;

            int true_face_id = paired_edges[edge_id];
            

            // bool v1_edge = compare_vertices(v1, v3, 1);
            // bool v2_edge = compare_vertices(v2, v3, 1);
            // // printf("%d %d %d\n", v1_edge, v2_edge, id);

            // bool or_v1_edge = compare_vertices(v1, v3, 0);
            // bool or_v2_edge = compare_vertices(v2, v3, 0);
            // printf("%d %d %d %d\n", or_v1_edge, or_v2_edge, v1_edge, v2_edge);

            int v1_2, v2_2, v3_1;
            getVerticesFromTriangleID(id, width, height, depth, v1_2, v2_2, v3_1);
            if(v1 != v1_2 && v2 != v1_2) v3_1 = v1_2;
            if(v1 != v2_2 && v2 != v2_2) v3_1 = v2_2;

            bool v1_edge = compare_vertices(v1, v3_1, 1);
            bool v2_edge = compare_vertices(v2, v3_1, 1);
            // printf("%d %d %d\n", v1_edge, v2_edge, id);

            bool or_v1_edge = compare_vertices(v1, v3_1, 0);
            bool or_v2_edge = compare_vertices(v2, v3_1, 0);
            // printf("%d %d %d %d\n", or_v1_edge, or_v2_edge, v1_edge, v2_edge);
            // printf("%.17f %.17f %.17f %.17f\n", decp_data[v3], decp_data[v3_1], input_data[v3], input_data[v3_1]);
            if(true_face_id != -1 && paired_faces[true_face_id] == edge_id)
            {
                // printf("2-saddle-4: %d\n", id);
                
                int true_face = true_face_id;
            
                int v1_1, v2_1, v3;
                getVerticesFromTriangleID(true_face, width, height, depth, v1_1, v2_1, v3);
                
                if(sortedV1 != v1_1 && sortedV2 != v1_1) v3 = v1_1;
                if(sortedV1 != v2_1 && sortedV2 != v2_1) v3 = v2_1;
                    
                double diff = get_delta(v3);
                // printf("%d %.17f %.17f %.17f %.17f %.17f %.17f\n", v_id, diff, dec_vertices[v_id].value, input_data[v_id]-dec_vertices[v_id].value,input_data[v_id] - bound, bound, range);
                double old_value = d_deltaBuffer[v3];
                
                if (true) {              
                    swap(v3, diff);
                }
                
            }
            
            else if(true_face_id >=0 && true_face_id < num_Vertices &&paired_vertices[true_face_id] == edge_id )
            {
                
                int v1_1, v2_1;
                getVertexIDsFromEdgeID(edge_id, width, height, depth, v1_1, v2_1);
                // printf("%d %d %d\n",face, v1_1, v2_1);
                int sortedV1 = !compare_vertices(v1_1, v2_1, 1)? v2_1:v1_1;
                int sortedV2 = !compare_vertices(v1_1, v2_1, 1)? v1_1:v2_1;
                
                int v1_2, v2_2, v3;
                getVerticesFromTriangleID(face, width, height, depth, v1_2, v2_2, v3);

                if(sortedV1 != v1_2 && sortedV2 != v1_2) v3 = v1_2;
                if(sortedV1 != v2_2 && sortedV2 != v2_2) v3 = v2_2;
                // double diff = abs(dec_vertices[v_id].value - (dec_vertices[v_id].value + input_data[v_id] - bound) /2.0);
                // if(diff>1e-16) dec_vertices[v_id].value = (dec_vertices[v_id].value + input_data[v_id] - bound) /2.0;
                // else dec_vertices[v_id].value = input_data[v_id] - bound;
                // printf("%d %d %d\n",face, sortedV1, sortedV2);
                // printf("%.17f %.17f %.17f %.17f %.17f %.17f\n",decp_data[sortedV1], decp_data[sortedV2], decp_data[v3], input_data[sortedV1], input_data[sortedV2], input_data[v3]);
                
                for(int v_id:{sortedV1, sortedV2})
                {
                    double diff = get_delta(v_id);
                    // printf("%d %.17f %.17f %.17f %.17f %.17f %.17f\n", v_id, diff, decp_data[v_id], input_data[v_id]-decp_data[v_id],input_data[v_id] - bound, bound, range);
                    double old_value = d_deltaBuffer[v_id];
                    
                    if (true) {              
                        swap(v_id, diff);
                    }
                }
                // if(decp_data[sortedV1] - (input_data[sortedV1] - bound) > 1e-16 * range) 
                // {
                //     // printf("%d\n", sortedV1);
                //     int v_id = sortedV1;
                    
                //     double diff = get_delta(v_id);
                //     // printf("%d %.17f %.17f %.17f %.17f %.17f %.17f\n", v_id, diff, decp_data[v_id], input_data[v_id]-decp_data[v_id],input_data[v_id] - bound, bound, range);
                //     double old_value = d_deltaBuffer[v_id];
                    
                //     if (true) {              
                //         swap(v_id, diff);
                //     }

                // }
                // else
                // {
                    
                //     int v_id = sortedV2;
                //     double diff = get_delta(v_id);
                //     // printf("%d %.17f %.17f \n", v_id, decp_data[v_id], input_data[v_id] - bound);
                //     // printf("%d %.17f %d %.17f %d %.17f\n", sortedV1, input_data[sortedV1], sortedV2, input_data[sortedV2], v3, input_data[v3]);
                //     // printf("%d %.17f %d %.17f %d %.17f\n", sortedV1, decp_data[sortedV1], sortedV2, decp_data[sortedV2], v3, decp_data[v3]);
                //     double old_value = d_deltaBuffer[v_id];
                //     if (true) {              
                //         swap(v_id, diff);
                //     }
                // }
            }

            else
            {
                
            }
            
            
            
        }
        // if face is paired with a tetra in the decompressed data/
        
        else if(dec_paired_faces[id] != -1 && dec_paired_faces[id] < num_Tetrahedras && dec_paired_tetrahedras[dec_paired_faces[id]] == id) {
            // printf("%d %d\n", dec_faces[id].paired_tetrahedra_id, faces[id].paired_tetrahedra_id);
            // printf("2-saddle 4:%d\n", id);
            int tetrahedra = dec_paired_faces[id];
            
            int v1, v2, v3;
            getVerticesFromTriangleID(face, width, height, depth, v1, v2, v3);
            int v1_1, v2_1, v3_1, v4;
            getVerticesFromTetrahedraID(tetrahedra, width, height, depth, v1_1, v2_1, v3_1, v4);
            if(v1_1!=v1 and v1_1!=v2 and v1_1 != v3) v4 = v1_1;
            else if(v2_1!=v1 and v2_1!=v2 and v2_1 != v3) v4 = v2_1;
            else if(v3_1!=v1 and v3_1!=v2 and v3_1 != v3) v4 = v3_1;
            
            
            
            int f1_id = getTriangleID(v1, v2, v4, width, height, depth);
            
            int f2_id = getTriangleID(v1, v3, v4, width, height, depth);
            
            int f3_id = getTriangleID(v2, v3, v4, width, height, depth);
            
            
            bool f1_face = compare_vertices(v3, v4, 1);
            bool f2_face = compare_vertices(v2, v4, 1);
            bool f3_face = compare_vertices(v1, v4, 1);

            int sortedV1 = v1, sortedV2 = v2, sortedV3 = v3;
            int sortedV1_a = v1, sortedV2_a = v2, sortedV3_a = v4;
            int sortedV1_b = v1, sortedV2_b = v3, sortedV3_b = v4;
            int sortedV1_c = v2, sortedV2_c = v3, sortedV3_c = v4;
            

            bool or_f1_face = compare_vertices(v3, v4, 0);
            bool or_f2_face = compare_vertices(v2, v4, 0);
            bool or_f3_face = compare_vertices(v1, v4, 0);
            
            if (decp_data[sortedV1] < decp_data[sortedV2] || (decp_data[sortedV1] == decp_data[sortedV2] && sortedV1 < sortedV2)) {
                int temp = sortedV1;
                sortedV1 = sortedV2;
                sortedV2 = temp;
            }
            if (decp_data[sortedV1] < decp_data[sortedV3] || (decp_data[sortedV1] == decp_data[sortedV3] && sortedV1 < sortedV3)) {
                int temp = sortedV1;
                sortedV1 = sortedV3;
                sortedV3 = temp;
            }
            if (decp_data[sortedV2] < decp_data[sortedV3] || (decp_data[sortedV2] == decp_data[sortedV3] && sortedV2 < sortedV3)) {
                int temp = sortedV2;
                sortedV2 = sortedV3;
                sortedV3 = temp;
            }
            
            if (decp_data[sortedV1_a] < decp_data[sortedV2_a] || (decp_data[sortedV1_a] == decp_data[sortedV2_a] && sortedV1_a < sortedV2_a)) {
                int temp = sortedV1_a;
                sortedV1_a = sortedV2_a;
                sortedV2_a = temp;
            }
            if (decp_data[sortedV1_a] < decp_data[sortedV3_a] || (decp_data[sortedV1_a] == decp_data[sortedV3_a] && sortedV1_a < sortedV3_a)) {
                int temp = sortedV1_a;
                sortedV1_a = sortedV3_a;
                sortedV3_a = temp;
            }
            if (decp_data[sortedV2_a] < decp_data[sortedV3_a] || (decp_data[sortedV2_a] == decp_data[sortedV3_a] && sortedV2_a < sortedV3_a)) {
                int temp = sortedV2_a;
                sortedV2_a = sortedV3_a;
                sortedV3_a = temp;
            }

            if (decp_data[sortedV1_b] < decp_data[sortedV2_b] || (decp_data[sortedV1_b] == decp_data[sortedV2_b] && sortedV1_b < sortedV2_b)) {
                int temp = sortedV1_b;
                sortedV1_b = sortedV2_b;
                sortedV2_b = temp;
            }
            if (decp_data[sortedV1_b] < decp_data[sortedV3_b] || (decp_data[sortedV1_b] == decp_data[sortedV3_b] && sortedV1_b < sortedV3_b)) {
                int temp = sortedV1_b;
                sortedV1_b = sortedV3_b;
                sortedV3_b = temp;
            }
            if (decp_data[sortedV2_b] < decp_data[sortedV3_b] || (decp_data[sortedV2_b] == decp_data[sortedV3_b] && sortedV2_b < sortedV3_b)) {
                int temp = sortedV2_b;
                sortedV2_b = sortedV3_b;
                sortedV3_b = temp;
            }

            if (decp_data[sortedV1_c] < decp_data[sortedV2_c] || (decp_data[sortedV1_c] == decp_data[sortedV2_c] && sortedV1_c < sortedV2_c)) {
                int temp = sortedV1_c;
                sortedV1_c = sortedV2_c;
                sortedV2_c = temp;
            }
            if (decp_data[sortedV1_c] < decp_data[sortedV3_c] || (decp_data[sortedV1_c] == decp_data[sortedV3_c] && sortedV1_c < sortedV3_c)) {
                int temp = sortedV1_c;
                sortedV1_c = sortedV3_c;
                sortedV3_c = temp;
            }
            if (decp_data[sortedV2_c] < decp_data[sortedV3_c] || (decp_data[sortedV2_c] == decp_data[sortedV3_c] && sortedV2_c < sortedV3_c)) {
                int temp = sortedV2_c;
                sortedV2_c = sortedV3_c;
                sortedV3_c = temp;
            }
            
            
                if(f1_face && f2_face && f3_face)
                {
                    if(f1_face!=or_f1_face)
                    {
                        
                        if(f1_face)
                        {
                        
                            
                            double diff = get_delta(v3);
                            double old_value = d_deltaBuffer[v1];
                            if (true) {              
                                swap(v3, diff);
                            }
                            
                        }
                        // edge < e1 in original data while edge > e1 in the decompressed data
                        else
                        {
                            
                            if(sortedV1 == sortedV1_a)
                            {
                                
                                // double diff = abs(decp_data[sortedV2_a]-(input_data[sortedV2_a] - bound + decp_data[sortedV2_a])/2.0);
                                // if(diff>1e-16) decp_data[sortedV2_a] = (input_data[sortedV2_a] - bound + decp_data[sortedV2_a])/2.0;
                                // else decp_data[sortedV2_a] = (input_data[sortedV2_a] - bound);
                                double diff = get_delta(sortedV2_a);
                                double old_value = d_deltaBuffer[sortedV2_a];
                                if (true) {              
                                    swap(sortedV2_a, diff);
                                }
                            }
                            else 
                            {
                                
                                if(decp_data[sortedV1] > decp_data[sortedV1_a]) {
                                    
                                    // double diff = abs(decp_data[sortedV1_a]-(input_data[sortedV1_a] - bound + decp_data[sortedV1_a])/2.0);
                                    // if(diff>1e-16) decp_data[sortedV1_a] = (input_data[sortedV1_a] - bound + decp_data[sortedV1_a])/2.0;
                                    // else decp_data[sortedV1_a] = (input_data[sortedV1_a] - bound);
                                    double diff = get_delta(sortedV1_a);
                                    double old_value = d_deltaBuffer[sortedV1_a];
                                    if (true) {              
                                        swap(sortedV1_a, diff);
                                    }   
                                }   
                                
                                
                            }
                        }
                    }
                    else if(f2_face!=or_f2_face)
                    {
                        // printf("2-saddle-6: %d %d %d %d %d %d\n", f1_face, f2_face, f3_face, or_f1_face, or_f2_face, or_f3_face);
                        // if edge<e1 in original data while edge>e1 in the decompressed data
                        if(f2_face)
                        {
                        
                            // if(sortedV1 == sortedV1_b) 
                            // {
                                
                            //     // double diff = abs(decp_data[sortedV2]-(input_data[sortedV2] - bound + decp_data[sortedV2])/2.0);
                            //     // if(diff>1e-16) decp_data[sortedV2] = (input_data[sortedV2] - bound + decp_data[sortedV2])/2.0;
                            //     // else decp_data[sortedV2] = (input_data[sortedV2] - bound);
                            //     double diff = (input_data[sortedV2] - bound - decp_data[sortedV2])/2.0;
                            //     double old_value = d_deltaBuffer[sortedV2];
                            //     if (true) {              
                            //         swap(sortedV2, diff);
                            //     }
                    
                            // }
                            // else 
                            // {
                            //     // v1_edge: v1小于edge。 ！or_v1_edge, v1>edge, 那就要减小edge，
                            //     if(decp_data[sortedV1_b] < decp_data[sortedV1]){
                                    
                            //         // double diff = abs(decp_data[sortedV1]-(input_data[sortedV1] - bound + decp_data[sortedV1])/2.0);
                            //         // if(diff>1e-16) decp_data[sortedV1] = (input_data[sortedV1] - bound + decp_data[sortedV1])/2.0;
                            //         // else decp_data[sortedV1] = (input_data[sortedV1] - bound);
                            //         // cout<<decp_data[sortedV1_b]->value<<","<<(input_data[sortedV1_b] - bound)<<endl;
                            //         double diff = get_delta(sortedV1);
                            //         double old_value = d_deltaBuffer[sortedV1];
                            //         if (true) {              
                            //             swap(sortedV1, diff);
                            //         }
                            //     }
                            //     // if(id==33000) cout<<v1_edge<<", "<<or_v1_edge<<endl;
                                
                                
                            // }
                            double diff = get_delta(v2);
                            double old_value = d_deltaBuffer[v2];
                            if (true) {              
                                swap(v2, diff);
                            }
                            
                        }
                        // edge < e1 in original data while edge > e1 in the decompressed data
                        else
                        {
                            
                            if(sortedV1 == sortedV1_b)
                            {
                                
                                // double diff = abs(decp_data[sortedV2_b]-(input_data[sortedV2_b] - bound + decp_data[sortedV2_b])/2.0);
                                // if(diff>1e-16) decp_data[sortedV2_b] = (input_data[sortedV2_b] - bound + decp_data[sortedV2_b])/2.0;
                                // else decp_data[sortedV2_b] = (input_data[sortedV2_b] - bound);
                                double diff = get_delta(sortedV2_b);
                                double old_value = d_deltaBuffer[sortedV2_b];
                                if (true) {              
                                    swap(sortedV2_b, diff);
                                }
                            }
                            else 
                            {
                                
                                if(decp_data[sortedV1] > decp_data[sortedV1_b]) {
                                    
                                    // double diff = abs(decp_data[sortedV1_b]-(input_data[sortedV1_b] - bound + decp_data[sortedV1_b])/2.0);
                                    // if(diff>1e-16) decp_data[sortedV1_b] = (input_data[sortedV1_b] - bound + decp_data[sortedV1_b])/2.0;
                                    // else decp_data[sortedV1_b] = (input_data[sortedV1_b] - bound);
                                    double diff = get_delta(sortedV1_b);
                                    double old_value = d_deltaBuffer[sortedV1_b];
                                    if (true) {              
                                        swap(sortedV1_b, diff);
                                    }   
                                }   
                                
                                
                            }
                        }
                    }

                    else if(f3_face!=or_f3_face)
                    {
                        // printf("2-saddle-7: %d %d %d %d %d %d\n", f1_face, f2_face, f3_face, or_f1_face, or_f2_face, or_f3_face);
                        // if edge<e1 in original data while edge>e1 in the decompressed data
                        if(f3_face)
                        {
                        
                            // if(sortedV1 == sortedV1_c) 
                            // {
                                
                            //     // double diff = abs(decp_data[sortedV2]-(input_data[sortedV2] - bound + decp_data[sortedV2])/2.0);
                            //     // if(diff>1e-16) decp_data[sortedV2] = (input_data[sortedV2] - bound + decp_data[sortedV2])/2.0;
                            //     // else decp_data[sortedV2] = (input_data[sortedV2] - bound);
                            //     double diff = (input_data[sortedV2] - bound - decp_data[sortedV2])/2.0;
                            //     double old_value = d_deltaBuffer[sortedV2];
                            //     if (true) {              
                            //         swap(sortedV2, diff);
                            //     }
                    
                            // }
                            // else 
                            // {
                            //     // v1_edge: v1小于edge。 ！or_v1_edge, v1>edge, 那就要减小edge，
                            //     if(decp_data[sortedV1_c] < decp_data[sortedV1]){
                                    
                            //         // double diff = abs(decp_data[sortedV1]-(input_data[sortedV1] - bound + decp_data[sortedV1])/2.0);
                            //         // if(diff>1e-16) decp_data[sortedV1] = (input_data[sortedV1] - bound + decp_data[sortedV1])/2.0;
                            //         // else decp_data[sortedV1] = (input_data[sortedV1] - bound);
                            //         // cout<<decp_data[sortedV1_c]->value<<","<<(input_data[sortedV1_c] - bound)<<endl;
                            //         double diff = get_delta(sortedV1);
                            //         double old_value = d_deltaBuffer[sortedV1];
                            //         if (true) {              
                            //             swap(sortedV1, diff);
                            //         }
                            //     }
                            //     // if(id==33000) cout<<v1_edge<<", "<<or_v1_edge<<endl;
                                
                                
                            // }
                            double diff = get_delta(v1);
                            double old_value = d_deltaBuffer[v1];
                            if (true) {              
                                swap(v1, diff);
                            }
                        }
                        // edge < e1 in original data while edge > e1 in the decompressed data
                        else
                        {
                            
                            if(sortedV1 == sortedV1_c)
                            {
                                
                                // double diff = abs(decp_data[sortedV2_c]-(input_data[sortedV2_c] - bound + decp_data[sortedV2_c])/2.0);
                                // if(diff>1e-16) decp_data[sortedV2_c] = (input_data[sortedV2_c] - bound + decp_data[sortedV2_c])/2.0;
                                // else decp_data[sortedV2_c] = (input_data[sortedV2_c] - bound);
                                double diff = get_delta(sortedV2_c);
                                double old_value = d_deltaBuffer[sortedV2_c];
                                if (true) {              
                                    swap(sortedV2_c, diff);
                                }
                            }
                            else 
                            {
                                
                                if(decp_data[sortedV1] > decp_data[sortedV1_c]) {
                                    
                                    // double diff = abs(decp_data[sortedV1_c]-(input_data[sortedV1_c] - bound + decp_data[sortedV1_c])/2.0);
                                    // if(diff>1e-16) decp_data[sortedV1_c] = (input_data[sortedV1_c] - bound + decp_data[sortedV1_c])/2.0;
                                    // else decp_data[sortedV1_c] = (input_data[sortedV1_c] - bound);
                                    double diff = get_delta(sortedV1_c);
                                    double old_value = d_deltaBuffer[sortedV1_c];
                                    if (true) {              
                                        swap(sortedV1_c, diff);
                                    }   
                                }   
                                
                                
                            }
                        }
                    }
                }
                

                // tetrahedra is paired with the cell in the second round, but shouldn't
                else
                {
                    // find if tetrahedra's paired cell is paired with any cell in the original data;
                    
                    int true_cell = paired_tetrahedras[tetrahedra];
                    // printf("2-saddle-5: %d %d %d %d %d %d %d %d\n", f1_face, f2_face, f3_face, or_f1_face, or_f2_face, or_f3_face, true_cell, num_Tetrahedras);
                    if(true_cell>=0 && true_cell<num_Tetrahedras)
                    {
                        for(int m:{sortedV1, sortedV2, sortedV3})
                        {
                            double diff = get_delta(m);
                            double old_value = d_deltaBuffer[m];
                            if (true) {              
                                swap(m, diff);
                            }
                        }
                    }
                    // if not paired with any cell in the original data;
                    else
                    {
                        
                        int id = get_smallest_vertex(v1, v2, v3, width, height, depth);
                        if(id!=-1)
                        {
                            int m =id;
                            double diff = get_delta(m);
                            double old_value = d_deltaBuffer[m];
                            if (true) {              
                                swap(m, diff);
                            }
                        }
                        else
                        {
                            for(int m:{sortedV1, sortedV2, sortedV3})
                            {
                                double diff = get_delta(m);
                                double old_value = d_deltaBuffer[m];
                                if (true) {              
                                    swap(m, diff);
                                }
                            }
                        }
                        
                    }
                    
                }
            

            
        }
    }
    
}

__global__ void fix_maximum(int width, int height, int depth){
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i>=count_f_max) return;
    
    int id = false_max[i];
    
    int face = id;
    // int id = face->id;
    // false positive case: not maximum in original data
    if(paired_tetrahedras[id]!=-1){
        // if( blockIdx.x == 0) printf("1-1: %d %d\n", i, threadIdx.x);
        // two cases: edge paired with another face, or edge is fp saddle itself, but already solved by the saddle process.
        // find its paired edge in the original/decompressed data.

        
        int face1 = paired_tetrahedras[id];
        
        // find the vertex in the true face that is not included in the false face. decrease the value of it.
        
        int v1, v2, v3;
        getVerticesFromTriangleID(face1, width, height, depth, v1, v2, v3);
        int v1_1, v2_1, v3_1, v4;
        getVerticesFromTetrahedraID(id, width, height, depth, v1_1, v2_1, v3_1, v4);

        
        if(v1_1!=v1 and v1_1!=v2 and v1_1 != v3) v4 = v1_1;
        else if(v2_1!=v1 and v2_1!=v2 and v2_1 != v3) v4 = v2_1;
        else if(v3_1!=v1 and v3_1!=v2 and v3_1 != v3) v4 = v3_1;
        
        
        bool f1_face = compare_vertices(v1, v4, 1);
        bool f2_face = compare_vertices(v2, v4, 1);
        bool f3_face = compare_vertices(v3, v4, 1);
        
    
        
        bool or_f1_face = compare_vertices(v1, v4, 0);
        bool or_f2_face = compare_vertices(v2, v4, 0);
        bool or_f3_face = compare_vertices(v3, v4, 0);
        
        
        // if(id == 27475)printf("1max: %d %d %d %d %d %d %d %d %d %d %d\n", v1, v2, v3, v4, f1_face, f2_face, f3_face, or_f1_face, or_f2_face, or_f3_face, id);
        if(or_f1_face == 1 && or_f2_face == 1 && or_f3_face == 1)
        {
            // printf("1-max: %d\n", id);
            double diff = get_delta(v4);
            double old_value = d_deltaBuffer[v4];
        
            if ( diff > old_value) 
            {                    
                swap(v4, diff);
            }
        }
        // id is paired with face in the second round
        else
        {
            // printf("f-max: %d\n", id);
            // if face1 is already be paired with another cell in the decompressed data, then face1 cant be paired with id, need to find this cell;
            int false_cell = dec_paired_faces[face1];
            
            if(false_cell>=0)
            {
                // if is paired with an edge
                if(false_cell < num_Edges && dec_paired_edges[false_cell] == face1)
                {
                    // printf("1-max-2: %d\n", id);
                    // if(id==27475) printf("%d\n", false_cell);
                    int v1, v2;
                    getVertexIDsFromEdgeID(false_cell, width, height, depth, v1, v2);
                    int sortedV1 = compare_vertices(v1, v2, 1)? v1:v2;
                    int sortedV2 = compare_vertices(v1, v2, 1)? v2:v1;
                    double diff = get_delta(sortedV1);
                    double old_value = d_deltaBuffer[sortedV1];
                    if (true) {              
                        swap(sortedV1, diff);
                    }
                }
                // if is paired with another tetra
                else if(false_cell < num_Tetrahedras && dec_paired_tetrahedras[false_cell] == face1)
                {
                    // printf("1-max-3: %d\n", id);
                    int v1, v2, v3;
                    getVerticesFromTriangleID(face1, width, height, depth, v1, v2, v3);
                    // int v1_1, v2_1, v3_1, v4;
                    // getVerticesFromTetrahedraID(false_cell, width, height, depth, v1_1, v2_1, v3_1, v4);

                    
                    // if(v1_1!=v1 and v1_1!=v2 and v1_1 != v3) v4 = v1_1;
                    // else if(v2_1!=v1 and v2_1!=v2 and v2_1 != v3) v4 = v2_1;
                    // else if(v3_1!=v1 and v3_1!=v2 and v3_1 != v3) v4 = v3_1;
                    int sortedV1, sortedV2, sortedV3;
                    sortVerticesByData(v1, v2, v3, sortedV1, sortedV2, sortedV3, 1);
                    
                    double diff = get_delta(v4);
                    double old_value = d_deltaBuffer[v4];
                
                    if ( diff > old_value) 
                    {                    
                        swap(v4, diff);
                    }
                }
            }

            // else not paired with any cell in the decp;
            
        }
        
        

    }

    // false negative case, face did not paired with any edge in original data, but paried with one edge in decompressed data
    else{
        
        // face should not be paired with anything in the input data.
        // find its paired edge in the dec;
        int face1 = dec_paired_tetrahedras[id];
        // printf("%d\n",dec_tetrahedras[id].paired_face_id);
        // this edge should be either paired with another face, or a saddle (will be solve by the saddle process)
        // only need to tackle the case that is paired with another face
        // find the edges paired face in the dec.
        
        // find face1's paired tetrahedra in the original data
        // if( blockIdx.x == 0 && face1!=-1 && paired_faces[face1]!= -1 && paired_faces[face1]>0 && paired_faces[face1]<num_Tetrahedras) printf("2-1: %d %d %d %d %d\n", i, threadIdx.x, face1, paired_faces[face1]);
        
        // if(face1!=-1 && paired_faces[face1]!=-1 && paired_faces[face1]>=0 && paired_faces[face1]<num_Tetrahedras && paired_tetrahedras[paired_faces[face1]] == face1)
        // {
            
        //     // if edge is paired with a face
            
        //     // if( blockIdx.x == 0) printf("2-2: %d %d %d %d %d\n", i, threadIdx.x, face1);
        //     int true_face = paired_faces[face1];
        //     int v1, v2, v3;
        //     getVerticesFromTriangleID(face1, width, height, depth, v1, v2, v3);
        //     int v1_1, v2_1, v3_1, v4;
        //     getVerticesFromTetrahedraID(true_face, width, height, depth, v1_1, v2_1, v3_1, v4);
            
        //     if(v1_1!=v1 and v1_1!=v2 and v1_1 != v3) v4 = v1_1;
        //     else if(v2_1!=v1 and v2_1!=v2 and v2_1 != v3) v4 = v2_1;
        //     else if(v3_1!=v1 and v3_1!=v2 and v3_1 != v3) v4 = v3_1;
            
            
        //     double diff = get_delta(v4);
        //     double old_value = d_deltaBuffer[v4];
        //     printf("2max: %d %d %d %d %d %.17f\n", v1, v2, v3, v4, diff);
        //     if ( diff > old_value) 
        //     {                    
        //         swap(v4, diff);
        //     }
            
        // }

        // did not paired with any face, then v1_edge and v2_edge should not be 1,1
        // else
        // {
            
            // printf("%d %d\n",id, tetrahedras[id].paired);
            // if( blockIdx.x == 0) printf("2: %d %d %d %d %d\n", i, threadIdx.x, face1);
            int v1, v2, v3;
            getVerticesFromTriangleID(face1, width, height, depth, v1, v2, v3);
            int v1_1, v2_1, v3_1, v4;
            getVerticesFromTetrahedraID(id, width, height, depth, v1_1, v2_1, v3_1, v4);
            
            
            if(v1_1!=v1 and v1_1!=v2 and v1_1 != v3) v4 = v1_1;
            else if(v2_1!=v1 and v2_1!=v2 and v2_1 != v3) v4 = v2_1;
            else if(v3_1!=v1 and v3_1!=v2 and v3_1 != v3) v4 = v3_1;
            

            int f1_id = getTriangleID(v1, v2, v4, width, height, depth);
            int f2_id = getTriangleID(v1, v3, v4, width, height, depth);
            int f3_id = getTriangleID(v2, v3, v4, width, height, depth);
            
            bool f1_face = compare_vertices(v1, v4, 1);
            bool f2_face = compare_vertices(v2, v4, 1);
            bool f3_face = compare_vertices(v3, v4, 1);
            
        
            
            bool or_f1_face = compare_vertices(v1, v4, 0);
            bool or_f2_face = compare_vertices(v2, v4, 0);
            bool or_f3_face = compare_vertices(v3, v4, 0);
            
            if(f1_face && f2_face && f3_face)
            {
                if(!or_f1_face && f1_face)
                {
                    // printf("2-max-1: %d %d %d %d %d %d %d\n", id, f1_face, f2_face, f3_face, or_f1_face, or_f2_face, or_f3_face);
                    
                    double diff = get_delta(v1);
                    double old_value = d_deltaBuffer[v1];
                    if (true) {              
                        swap(v1, diff);
                    }
                }

                if(!or_f2_face && f2_face)
                {
                    // printf("2-max-2: %d\n", id);
                    double diff = get_delta(v2);
                    double old_value = d_deltaBuffer[v2];
                    if (true) {              
                        swap(v2, diff);
                    }
                }

                if(!or_f3_face && f3_face)
                {
                    // printf("2-max-3: %d\n", id);
                    double diff = get_delta(v3);
                    double old_value = d_deltaBuffer[v3];
                    if (true) {              
                        swap(v3, diff);
                    }
                }

                if(or_f1_face && or_f2_face && or_f3_face)
                {
                    // printf("false here:%d\n", id);
                    int it = paired_faces[face1];
                    if(it!=-1)
                    {
                        int v1_1, v2_1, v3_1, v4;
                        getVerticesFromTetrahedraID(it, width, height, depth, v1_1, v2_1, v3_1, v4);
                        
                        
                        if(v1_1!=v1 and v1_1!=v2 and v1_1 != v3) v4 = v1_1;
                        else if(v2_1!=v1 and v2_1!=v2 and v2_1 != v3) v4 = v2_1;
                        else if(v3_1!=v1 and v3_1!=v2 and v3_1 != v3) v4 = v3_1;
                        double diff = get_delta(v4);
                        double old_value = d_deltaBuffer[v4];
                        if (true) {              
                            swap(v4, diff);
                        }
                    }
                }
            }
            
            
            else
            {
                
                if(paired_faces[face1]!=-1)
                {
                    int paired_id = paired_faces[face1];
                    if(paired_id >=0 && paired_id < num_Edges && paired_edges[paired_id] == face1)
                    {
                        int e_v1, e_v2;
                        getVertexIDsFromEdgeID(paired_id,width, height, depth, e_v1, e_v2);
                        for(int m:{e_v1, e_v2})
                        {
                            double diff = get_delta(m);
                            double old_value = d_deltaBuffer[m];
                            if (true) {              
                                swap(m, diff);
                            } 
                        }
                    }
                    else
                    {
                        int e_v1, e_v2, e_v3, e_v4;
                        getVerticesFromTetrahedraID(paired_id,width, height, depth, e_v1, e_v2, e_v3, e_v4);
                        for(int m:{e_v1, e_v2,e_v3, e_v4})
                        {
                            double diff = get_delta(m);
                            double old_value = d_deltaBuffer[m];
                            if (true) {              
                                swap(m, diff);
                            } 
                        }
                    }
                }
                
                else
                {   
                    printf("2-max-4: %d %d %d %d %d %d %d\n", f1_face, f2_face, f3_face, or_f1_face, or_f2_face, or_f3_face, paired_faces[face1]);
                    int sortedV2 = v2;
                    int sortedV3 = v3;
                    if(!or_f1_face)
                    {
                        int m = v1;
                        double diff = get_delta(m);
                        double old_value = d_deltaBuffer[m];
                        if (true) {              
                            swap(m, diff);
                        } 
                    }
                    if(!or_f2_face)
                    {
                        int m = v2;
                        double diff = get_delta(m);
                        double old_value = d_deltaBuffer[m];
                        if (true) {              
                            swap(m, diff);
                        } 
                    }
                    if(!or_f3_face)
                    {
                        int m = v3;
                        double diff = get_delta(m);
                        double old_value = d_deltaBuffer[m];
                        if (true) {              
                            swap(m, diff);
                        } 
                    }
                    

                }
                

                
            } 
        }

        

    // }
}

__global__ void init_vertices(int type =0)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=num_Vertices) return;
    if(type == 1)
    {
        paired_vertices[i]=-1;
    }
    dec_paired_vertices[i]=-1;
    
}

__global__ void init_edges(int type =0)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=num_Edges) return;
    if(type==1)
    {
        paired_edges[i]=-1;
    }
    dec_paired_edges[i]= -1;

}

__global__ void init_faces(int type =0)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=num_Faces) return;
    if(type==1)
    {
        paired_faces[i]=-1;
    }
    dec_paired_faces[i] = -1;
    
}

__global__ void init_masks(int type =0)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=num_Faces) return;
    visited_faces[i] = -1;

    
}

__global__ void init_tetras(int type =0)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=num_Tetrahedras) return;
    if(type==1)
    {
        paired_tetrahedras[i]=-1;
    }
    dec_paired_tetrahedras[i] = -1;
    
    
}

__device__ void get_cell_tetra(int face_id, int width, int height, int depth, int* x, int* y)
{
    int v1, v2, v3;
    
    int num =0;
    getVerticesFromTriangleID(face_id, width, height, depth, v1, v2, v3);
    int v1_x = v1 % width;
    int v1_y = (v1 / (width)) % height;
    int v1_z = (v1 / (width * height)) % depth;

    int v2_x = v2 % width;
    int v2_y = (v2 / (width)) % height;
    int v2_z = (v2 / (width * height)) % depth;

    int v3_x = v3 % width;
    int v3_y = (v3 / (width)) % height;
    int v3_z = (v3 / (width * height)) % depth;
    for (int i = 0; i < 14; i++)
    {
        int nx = v1_x + directions1[i][0];
        int ny = v1_y + directions1[i][1];
        int nz = v1_z + directions1[i][2];
        int neighbor_id1 = nx + ny * width + nz * width * height;
        
        if(nx >= width || ny >= height || nz >= depth || nx < 0 ||
           ny < 0 || nz < 0 || neighbor_id1 >= num_Vertices || neighbor_id1 < 0 ) continue;

        for (int j = 0; j < 14; j++)
        {
            
            int nx1 = v2_x + directions1[j][0];
            int ny1 = v2_y + directions1[j][1];
            int nz1 = v2_z + directions1[j][2];
            int neighbor_id2 = nx1 + ny1 * width + nz1 * width * height;
            if(nx1 >= width || ny1 >= height || nz1 >= depth || nx1 < 0 ||
                ny1 < 0 || nz1 < 0 || neighbor_id2 >= num_Vertices || neighbor_id2 < 0 ) continue;
            for (int k = 0; k < 14; k++) {

                int nx2 = v3_x + directions1[k][0];
                int ny2 = v3_y + directions1[k][1];
                int nz2 = v3_z + directions1[k][2];
                int neighbor_id3 = nx2 + ny2 * width + nz2 * width * height;
                if(nx2 >= width || ny2 >= height || nz2 >= depth || nx2 < 0 ||
                    ny2 < 0 || nz2 < 0 || neighbor_id3 >= num_Vertices || neighbor_id3 < 0 ) continue;
                
                if(!(neighbor_id1 == neighbor_id2 && neighbor_id2 == neighbor_id3 && neighbor_id1 == neighbor_id3 )) continue;
                if (neighbor_id1 == v1 || neighbor_id1 == v2 || neighbor_id1== v3) continue;

                int v4 = neighbor_id1;
                
                int tetra_id = getTetrahedraID(v1, v2, v3, v4, width, height, depth);
                if(num==0)
                {
                    *x = tetra_id;
                    num++;
                }
                else
                {
                    *y = tetra_id;
                }
                
            }
        }
    }

    return;

}

__device__ void get_cell_face()
{

}

__device__ bool is_false(int id, int width, int height, int depth)
{
    int t;
    int dec_t;
    int current_id = id;
    int dec_current_id = id;
    int old_id;
    bool f_sign = false;
    

    t = id;
    dec_t = id;
    current_id = id;
    dec_current_id = id;
    if(t!=dec_t)
    {
        f_sign = true;
        return f_sign;
    }
    do
    {
        
        old_id = current_id;
        t = paired_tetrahedras[current_id];
        
        dec_t = dec_paired_tetrahedras[current_id];

        if(t==-1)
        { 
            break;
        }

        if(t != dec_t) 
        {
            if(id==3827) printf("false_as_vpath: %d %d %d %d\n", id, current_id, t, dec_t);
            
            f_sign = true;
            int idx_fp_saddle = atomicAdd(&count_f_as_vpath, 1);// in one instruction
            false_as_vpath[idx_fp_saddle] = id;
            false_as_vpath_index[2*id] = current_id;
            false_as_vpath_index[2*id + 1] = 3;
            return f_sign;
            
        }

        int connectedFace = t;
        int dec_connectedFace = dec_t;
        
        
        if(paired_faces[t] == -1 || (paired_faces[t] != -1 && paired_tetrahedras[paired_faces[t]] != t))
        {
            break;
        }
        int t1 = -1;
        int t2 = -1;
        get_cell_tetra(t, width, height, depth, &t1, &t2);
        current_id = t1 == current_id?t2:t1;
    } while(current_id!=old_id and current_id != -1);

    return f_sign; 
}

__device__ int get_paired_id(int id, int width, int height, int depth, int type = 0)
{
    int t = id;
    int current_id = id;
    int old_id;
    bool f_sign = false;
    int *paired_tetrahedras1 = paired_tetrahedras;
    int *paired_faces1 = paired_faces;
    if(type==1)
    {
        paired_tetrahedras1 = dec_paired_tetrahedras;
        paired_faces1 = dec_paired_faces;
    }
    do
    {
        
        old_id = current_id;
        t = paired_tetrahedras1[current_id];

        if(t==-1)
        { 
            break;
        }

        int connectedFace = t;
        
        
        if(paired_faces1[t] == -1 || (paired_faces1[t] != -1 && paired_tetrahedras1[paired_faces1[t]] != t))
        {
            break;
        }
        int t1 = -1;
        int t2 = -1;
        get_cell_tetra(t, width, height, depth, &t1, &t2);
        current_id = t1 == current_id?t2:t1;
    } while(current_id!=old_id and current_id != -1);

    return current_id; 
}


__global__ void get_false_as_vpath(int width, int height, int depth)
{
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=num_Faces) return;
    if(paired_faces[i]!=-1) return;
    int t;
    int dec_t;
    int current_id = i;
    int dec_current_id = i;
    int old_id;
    bool f_sign = false;
    int v1, v2, v3;
    
    int num =0;
    getVerticesFromTriangleID(i, width, height, depth, v1, v2, v3);
    int v1_x = v1 % width;
    int v1_y = (v1 / (width)) % height;
    int v1_z = (v1 / (width * height)) % depth;

    int v2_x = v2 % width;
    int v2_y = (v2 / (width)) % height;
    int v2_z = (v2 / (width * height)) % depth;

    int v3_x = v3 % width;
    int v3_y = (v3 / (width)) % height;
    int v3_z = (v3 / (width * height)) % depth;
    for (int i = 0; i < 14; i++) 
    {
        int nx = v1_x + directions1[i][0];
        int ny = v1_y + directions1[i][1];
        int nz = v1_z + directions1[i][2];
        int neighbor_id1 = nx + ny * width + nz * width * height;
        
        if(nx >= width || ny >= height || nz >= depth || nx < 0 ||
           ny < 0 || nz < 0 || neighbor_id1 >= num_Vertices || neighbor_id1 < 0 ) continue;

        for (int j = 0; j < 14; j++)
        {
            
            int nx1 = v2_x + directions1[j][0];
            int ny1 = v2_y + directions1[j][1];
            int nz1 = v2_z + directions1[j][2];
            int neighbor_id2 = nx1 + ny1 * width + nz1 * width * height;
            if(nx1 >= width || ny1 >= height || nz1 >= depth || nx1 < 0 ||
                ny1 < 0 || nz1 < 0 || neighbor_id2 >= num_Vertices || neighbor_id2 < 0 ) continue;
            for (int k = 0; k < 14; k++) {

                int nx2 = v3_x + directions1[k][0];
                int ny2 = v3_y + directions1[k][1];
                int nz2 = v3_z + directions1[k][2];
                int neighbor_id3 = nx2 + ny2 * width + nz2 * width * height;
                if(nx2 >= width || ny2 >= height || nz2 >= depth || nx2 < 0 ||
                    ny2 < 0 || nz2 < 0 || neighbor_id3 >= num_Vertices || neighbor_id3 < 0 ) continue;
                
                if(!(neighbor_id1 == neighbor_id2 && neighbor_id2 == neighbor_id3 && neighbor_id1 == neighbor_id3 )) continue;
                if (neighbor_id1 == v1 || neighbor_id1 == v2 || neighbor_id1== v3) continue;

                int v4 = neighbor_id1;
                
                int tetra_id = getTetrahedraID(v1, v2, v3, v4, width, height, depth);
                if(is_false(tetra_id, width, height, depth)) return;
                

                
            }
        }
    }
    
    
}


__global__ void get_orginal_saddle_min_pairs(int width, int height, int depth)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=num_Edges) return;
    if(paired_edges[i]!=-1) return;

    // descending, still from edges;
    int v1, v2;
    getVertexIDsFromEdgeID(i, width, height, depth, v1, v2);
    int v = v1;
    int dec_v = v1;

    int current_id = v1;
    int dec_current_id = v1;
    int connectedEdge;
    

    bool f_sign = false;
    do
    {
        
        v = current_id;

        if(paired_vertices[v]==-1)
        {
            break;
        }

        connectedEdge = paired_vertices[v];
        
        if(paired_edges[connectedEdge] == -1) 
        {
            break;
        }

        int v1_1, v2_1;
        getVertexIDsFromEdgeID(connectedEdge, width, height, depth, v1_1, v2_1);
        current_id = v1_1 == current_id?v2_1:v1_1;
    } while(paired_vertices[v]!=-1);


    connected_min_saddle[i*2] = v;

    v = v2;
    current_id = v2;

    f_sign = false;
    do
    {
        
        v = current_id;
        

        if(paired_vertices[v]==-1)
        {
            break;
        }

        connectedEdge = paired_vertices[v];

        if(paired_edges[connectedEdge] == -1) 
        {
            break;
        }

        int v1_1, v2_1;
        getVertexIDsFromEdgeID(connectedEdge, width, height, depth, v1_1, v2_1);
        current_id = v1_1 == current_id?v2_1:v1_1;
        
    } while(paired_vertices[v]!=-1);
    
    connected_min_saddle[i*2+1] = v;

    return;
}

__global__ void get_orginal_saddle_max_pairs(int width, int height, int depth)
{
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=num_Faces) return;
    if(paired_faces[i]!=-1) return;
    int t;
    int current_id = i;
    int old_id;
    bool f_sign = false;
    int v1, v2, v3;
    
    int num =0;
    getVerticesFromTriangleID(i, width, height, depth, v1, v2, v3);
    int v1_x = v1 % width;
    int v1_y = (v1 / (width)) % height;
    int v1_z = (v1 / (width * height)) % depth;

    int v2_x = v2 % width;
    int v2_y = (v2 / (width)) % height;
    int v2_z = (v2 / (width * height)) % depth;

    int v3_x = v3 % width;
    int v3_y = (v3 / (width)) % height;
    int v3_z = (v3 / (width * height)) % depth;
    int paired_num = 0;
    for (int i = 0; i < 14; i++) 
    {
        int nx = v1_x + directions1[i][0];
        int ny = v1_y + directions1[i][1];
        int nz = v1_z + directions1[i][2];
        int neighbor_id1 = nx + ny * width + nz * width * height;
        
        if(nx >= width || ny >= height || nz >= depth || nx < 0 ||
           ny < 0 || nz < 0 || neighbor_id1 >= num_Vertices || neighbor_id1 < 0 ) continue;

        for (int j = 0; j < 14; j++)
        {
            
            int nx1 = v2_x + directions1[j][0];
            int ny1 = v2_y + directions1[j][1];
            int nz1 = v2_z + directions1[j][2];
            int neighbor_id2 = nx1 + ny1 * width + nz1 * width * height;
            if(nx1 >= width || ny1 >= height || nz1 >= depth || nx1 < 0 ||
                ny1 < 0 || nz1 < 0 || neighbor_id2 >= num_Vertices || neighbor_id2 < 0 ) continue;
            for (int k = 0; k < 14; k++) {

                int nx2 = v3_x + directions1[k][0];
                int ny2 = v3_y + directions1[k][1];
                int nz2 = v3_z + directions1[k][2];
                int neighbor_id3 = nx2 + ny2 * width + nz2 * width * height;
                if(nx2 >= width || ny2 >= height || nz2 >= depth || nx2 < 0 ||
                    ny2 < 0 || nz2 < 0 || neighbor_id3 >= num_Vertices || neighbor_id3 < 0 ) continue;
                
                if(!(neighbor_id1 == neighbor_id2 && neighbor_id2 == neighbor_id3 && neighbor_id1 == neighbor_id3 )) continue;
                if (neighbor_id1 == v1 || neighbor_id1 == v2 || neighbor_id1== v3) continue;

                int v4 = neighbor_id1;
                
                int tetra_id = getTetrahedraID(v1, v2, v3, v4, width, height, depth);
                int paired_maximum = get_paired_id(tetra_id, width, height, depth);
                connected_max_saddle[2*i+paired_num] = paired_maximum;
                
                paired_num++;
                
            }
        }
    }
    
    
}

__global__ void get_false_as_vpath_connectivity(int width, int height, int depth)
{
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=num_Faces) return;
    if(dec_paired_faces[i]!=-1) return;
    int t;
    
    int current_id = i;
    
    int old_id;
    bool f_sign = false;
    int v1, v2, v3;
    
    int num =0;
    getVerticesFromTriangleID(i, width, height, depth, v1, v2, v3);
    int v1_x = v1 % width;
    int v1_y = (v1 / (width)) % height;
    int v1_z = (v1 / (width * height)) % depth;

    int v2_x = v2 % width;
    int v2_y = (v2 / (width)) % height;
    int v2_z = (v2 / (width * height)) % depth;

    int v3_x = v3 % width;
    int v3_y = (v3 / (width)) % height;
    int v3_z = (v3 / (width * height)) % depth;
    for (int i = 0; i < 14; i++) 
    {
        int nx = v1_x + directions1[i][0];
        int ny = v1_y + directions1[i][1];
        int nz = v1_z + directions1[i][2];
        int neighbor_id1 = nx + ny * width + nz * width * height;
        
        if(nx >= width || ny >= height || nz >= depth || nx < 0 ||
           ny < 0 || nz < 0 || neighbor_id1 >= num_Vertices || neighbor_id1 < 0 ) continue;

        for (int j = 0; j < 14; j++)
        {
            
            int nx1 = v2_x + directions1[j][0];
            int ny1 = v2_y + directions1[j][1];
            int nz1 = v2_z + directions1[j][2];
            int neighbor_id2 = nx1 + ny1 * width + nz1 * width * height;
            if(nx1 >= width || ny1 >= height || nz1 >= depth || nx1 < 0 ||
                ny1 < 0 || nz1 < 0 || neighbor_id2 >= num_Vertices || neighbor_id2 < 0 ) continue;
            for (int k = 0; k < 14; k++) {

                int nx2 = v3_x + directions1[k][0];
                int ny2 = v3_y + directions1[k][1];
                int nz2 = v3_z + directions1[k][2];
                int neighbor_id3 = nx2 + ny2 * width + nz2 * width * height;
                if(nx2 >= width || ny2 >= height || nz2 >= depth || nx2 < 0 ||
                    ny2 < 0 || nz2 < 0 || neighbor_id3 >= num_Vertices || neighbor_id3 < 0 ) continue;
                
                if(!(neighbor_id1 == neighbor_id2 && neighbor_id2 == neighbor_id3 && neighbor_id1 == neighbor_id3 )) continue;
                if (neighbor_id1 == v1 || neighbor_id1 == v2 || neighbor_id1== v3) continue;

                int v4 = neighbor_id1;
                
                int tetra_id = getTetrahedraID(v1, v2, v3, v4, width, height, depth);
                int dec_id = get_paired_id(tetra_id, width, height, depth, 1);
                if(dec_id != connected_max_saddle[i*2+num])
                {
                    if(is_false(tetra_id, width, height, depth)) return;
                }
                num++;

                
            }
        }
    }
    
    
}

__global__ void get_false_ds_vpath(int width, int height, int depth)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=num_Edges or paired_edges[i]!=-1) return;
    
    // descending, still from edges;
    int v1, v2;
    getVertexIDsFromEdgeID(i, width, height, depth, v1, v2);
    int v = v1;
    int dec_v = v1;

    int current_id = v1;
    int dec_current_id = v1;
    int connectedEdge;
    

    bool f_sign = false;
    do
    {
        
        v = current_id;
        
        dec_v = dec_current_id;

        if(v != dec_v) 
        {
            
            f_sign = true;
            break;
            
            
        }

        if(paired_vertices[v]==-1)
        {
            break;
        }

        connectedEdge = paired_vertices[v];
        
        int dec_connectedEdge = dec_paired_vertices[v];
        
        if(connectedEdge != dec_connectedEdge) 
        {
            int idx_fp_saddle = atomicAdd(&count_f_ds_vpath, 1);// in one instruction
            
            false_ds_vpath[idx_fp_saddle] = i;
            false_ds_vpath_index[i*2] = v;
            false_ds_vpath_index[i*2+1] = 0;
            return;
        }

        if(paired_edges[connectedEdge] == -1) 
        {
            break;
        }

        int v1_1, v2_1;
        getVertexIDsFromEdgeID(connectedEdge, width, height, depth, v1_1, v2_1);
        current_id = v1_1 == current_id?v2_1:v1_1;
    } while(paired_vertices[v]!=-1 and dec_paired_vertices[v]!=-1);


    

    v = v2;
    dec_v = v2;
    current_id = v2;
    dec_current_id = v2;


    f_sign = false;
    do
    {
        
        v = current_id;
        
        dec_v = dec_current_id;

        if(v != dec_v) 
        {
            
            f_sign = true;
            break;
            
            
        }

        if(paired_vertices[v]==-1)
        {
            break;
        }

        connectedEdge = paired_vertices[v];
        
        int dec_connectedEdge = dec_paired_vertices[v];
        
        if(connectedEdge != dec_connectedEdge) 
        {
            int idx_fp_saddle = atomicAdd(&count_f_ds_vpath, 1);// in one instruction
            
            false_ds_vpath[idx_fp_saddle] = i;
            false_ds_vpath_index[i*2] = v;
            false_ds_vpath_index[i*2+1] = 0;
            return;
        }

        if(paired_edges[connectedEdge] == -1) 
        {
            break;
        }

        int v1_1, v2_1;
        getVertexIDsFromEdgeID(connectedEdge, width, height, depth, v1_1, v2_1);
        current_id = v1_1 == current_id?v2_1:v1_1;
        
    } while(paired_vertices[v]!=-1 and dec_paired_vertices[v]!=-1);
    
    
    

    return;
}

__global__ void get_false_ds_vpath_connectivity(int width, int height, int depth)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=num_Edges or paired_edges[i]!=-1) return;
    
    // descending, still from edges;
    int v1, v2;
    getVertexIDsFromEdgeID(i, width, height, depth, v1, v2);
    int v = v1;

    int current_id = v1;
    int connectedEdge;
    
    int trouble_maker = -1;
    bool f_sign = false;
    do
    {
        
        v = current_id;

        if(dec_paired_vertices[v]==-1)
        {
            break;
        }

        connectedEdge = paired_vertices[v];
        
        int dec_connectedEdge = dec_paired_vertices[v];
        
        if(connectedEdge != dec_connectedEdge) 
        {
            if(trouble_maker == -1) trouble_maker = v;
        }

        if(dec_paired_edges[dec_connectedEdge] == -1) 
        {
            break;
        }

        int v1_1, v2_1;
        getVertexIDsFromEdgeID(dec_connectedEdge, width, height, depth, v1_1, v2_1);
        current_id = v1_1 == current_id?v2_1:v1_1;
    } while(dec_paired_vertices[v]!=-1);

    if(v != connected_min_saddle[i*2])
    {
        int idx_fp_saddle = atomicAdd(&count_f_ds_vpath, 1);// in one instruction
            
        false_ds_vpath[idx_fp_saddle] = i;
        false_ds_vpath_index[i*2] = trouble_maker;
        false_ds_vpath_index[i*2+1] = 0;
        return;
    }
    

    v = v2;
    current_id = v2;


    f_sign = false;
    do
    {
        
        v = current_id;

        if(dec_paired_vertices[v]==-1)
        {
            break;
        }

        connectedEdge = paired_vertices[v];
        
        int dec_connectedEdge = dec_paired_vertices[v];
        
        if(connectedEdge != dec_connectedEdge) 
        {
            if(trouble_maker == -1) trouble_maker = v;
        }

        if(dec_paired_edges[dec_connectedEdge] == -1) 
        {
            break;
        }

        int v1_1, v2_1;
        getVertexIDsFromEdgeID(dec_connectedEdge, width, height, depth, v1_1, v2_1);
        current_id = v1_1 == current_id?v2_1:v1_1;
    } while(dec_paired_vertices[v]!=-1);
    
    if(v != connected_min_saddle[i*2+1])
    {
        int idx_fp_saddle = atomicAdd(&count_f_ds_vpath, 1);// in one instruction
            
        false_ds_vpath[idx_fp_saddle] = i;
        false_ds_vpath_index[i*2] = trouble_maker;
        false_ds_vpath_index[i*2+1] = 0;
        return;
    }
    

    return;
}

__global__ void get_false_ds_wall(int width, int height, int depth, int* global_memory, int global_capacity) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=num_Faces or paired_faces[i]!=-1) return;
    
    int f = i;
    int dec_f = i;
    
   extern __shared__ int shared_memory[];  
    int shared_capacity = blockDim.x; // Assuming the shared memory is allocated for each block

    // Create a SimpleStack object
    SimpleStack stack(shared_memory, shared_capacity, global_memory, global_capacity);

    stack.push(i); // Initial push

    int triangleId = i;
    
    int false_index;
    
    while (!stack.isEmpty()) 
    {
        
        triangleId = stack.pop();

        if(visited_faces[triangleId] == -1)
        {
            visited_faces[triangleId] = 1;
            int v1, v2, v3;
            getVerticesFromTriangleID(triangleId, width, height, depth, v1, v2, v3);
            int v1_x = v1 % width;
            int v1_y = (v1 / (width)) % height;
            int v1_z = (v1 / (width * height)) % depth;

            int v2_x = v2 % width;
            int v2_y = (v2 / (width)) % height;
            int v2_z = (v2 / (width * height)) % depth;

            int v3_x = v3 % width;
            int v3_y = (v3 / (width)) % height;
            int v3_z = (v3 / (width * height)) % depth;
            int e1_id = getEdgeID(v1_x, v1_y, v1_z, v2_x, v2_y, v2_z, width, height, depth);
            int e2_id = getEdgeID(v1_x, v1_y, v1_z, v3_x, v3_y, v3_z, width, height, depth);
            int e3_id = getEdgeID(v2_x, v2_y, v2_z, v3_x, v3_y, v3_z, width, height, depth);
            for(int j:{e1_id, e2_id, e3_id})
            {   
                // if(j == -1 or j>num_Edges) printf("%d\n", triangleId);
                if(paired_edges[j]==-1)
                {
                    continue;
                }
                if(paired_edges[j]!=dec_paired_edges[j])
                {
                    int idx_fp_saddle = atomicAdd(&count_f_ds_wall, 1);// in one instruction
                    false_ds_wall[idx_fp_saddle] = i;
                    false_ds_wall_index[i] = j;
                    
                    return;
                }
                else
                {
                    int pairedCellId = paired_edges[j];
                    if(pairedCellId != triangleId and pairedCellId>=0 and pairedCellId < num_Faces and paired_faces[pairedCellId] == j) stack.push(pairedCellId);
                    
                }
            }
        }
    }
    
    
}

__global__ void fix_as_vpath(int width, int height, int depth)
{   
        
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=count_f_as_vpath) return;
    
    int id = false_as_vpath[i];
    

    int saddle = id;
    // fix saddle_max
    bool f_sign = false;
    
    if(false_as_vpath_index[2*id+1] == 2)
    {
        printf("type1: %d\n", id);
        int f_idx = false_as_vpath_index[2*id];
        
        // the edge's paired face is wrong, find the true_paired_face and false_paired_face, decrease the value of the true_paired_face.
        int face = f_idx;
        // printf("paired_face:%d %d %d %d %d %d\n", id, edge_idx, edges[edge_idx].paired_face_id, dec_edges[edge_idx].paired_face_id);
        int true_paired_tetra = paired_faces[f_idx];
        // auto false_paired_face = dec_faces[dec_edges[edge_idx].paired_face_id];

       
        int v1, v2, v3, v4;
        getVerticesFromTetrahedraID(true_paired_tetra, width, height, depth, v1, v2, v3,v4);
        int true_v3 = v4;

        
        if(v1!=v1 and v1!=v2 and v1 != v3) true_v3 = v1;
        else if(v2!=v1 and v2!=v2 and v2 != v3) true_v3 = v2;
        else if(v3!=v1 and v3!=v2 and v3 != v3) true_v3 = v3;
    
        double diff = get_delta(true_v3);
    
        double old_value = d_deltaBuffer[true_v3];
        if ( diff > old_value) 
        {                    
            swap(true_v3, diff);
        }


    }
    else
    {
        // printf("type2: %d\n", id);
        // the tetra's paired face is wrong
        int t_idx = false_as_vpath_index[2*id];
        int true_edge = paired_tetrahedras[t_idx];
        int false_edge = dec_paired_tetrahedras[t_idx];

        // find the false_face's paired tetra in the original data
        if(paired_faces[false_edge] >=0 && paired_faces[false_edge]< num_Tetrahedras && paired_tetrahedras[paired_faces[false_edge]] == false_edge)
        {
            

            int true_face = paired_faces[false_edge];
            int v1, v2, v3;
            getVerticesFromTriangleID(false_edge, width, height, depth, v1, v2, v3);
            int v1_1, v2_1, v3_1, v4;
            getVerticesFromTetrahedraID(true_face, width, height, depth, v1_1, v2_1, v3_1,v4);
            int true_v3 = v4;

            
            if(v1_1!=v1 and v1_1!=v2 and v1_1 != v3) true_v3 = v1_1;
            else if(v2_1!=v1 and v2_1!=v2 and v2_1 != v3) true_v3 = v2_1;
            else if(v3_1!=v1 and v3_1!=v2 and v3_1 != v3) true_v3 = v3_1;
            


        

            double diff = get_delta(true_v3);
    
            double old_value = d_deltaBuffer[true_v3];
            if (true) {              
                swap(true_v3, diff);
            } 
            
            
        }

        else
        {
            
            // printf("type2: %d %d %d %d\n", id, t_idx, false_edge, true_edge);
            int v1, v2, v3;
            getVerticesFromTriangleID(true_edge, width, height, depth, v1, v2, v3);
            int v1_1, v2_1, true_v3;
            getVerticesFromTriangleID(false_edge, width, height, depth, v1_1, v2_1, true_v3);

            if(v1_1!=v1 and v1_1!=v2 and v1_1!=v3) true_v3 = v1_1;
            else if(v2_1!=v1 and v2_1!=v2 and v2_1!=v3) true_v3 = v2_1;
            

            double diff = get_delta(true_v3);
    
            double old_value = d_deltaBuffer[true_v3];
            if (true) {              
                swap(true_v3, diff);
            } 


        }
        
    }
    // ascending, only face,
    return ;
}

__global__ void fix_ds_vpath(int width, int height, int depth)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=count_f_ds_vpath) return;
    int id = false_ds_vpath[i];
    int saddle = id;
    // fix saddle_max
    bool f_sign = false;
    if(false_ds_vpath_index[2*id+1]==1)
    {
        int edge_idx = false_ds_vpath_index[2*id];
        
        // the edge's paired face is wrong, find the true_paired_face and false_paired_face, decrease the value of the true_paired_face.
            
        int true_paired_vertex = paired_edges[edge_idx];
        
        
        double diff = get_delta(true_paired_vertex);
        
        double old_value = d_deltaBuffer[true_paired_vertex];
        if ( diff > old_value) { 
                        
            swap(true_paired_vertex, diff);
        } 

            
    }

    else
    {
        int v_idx = false_ds_vpath_index[2*id];  
        // printf("%d %d %d\n", v_idx, vertices[v_idx].paired_edge_id, vertices[v_idx].paired);
        int true_paired_edge = paired_vertices[v_idx];
        int false_paired_edge = dec_paired_vertices[v_idx];
        
        int v1, v2;
        getVertexIDsFromEdgeID(true_paired_edge, width, height, depth, v1, v2);

        auto true_vertex = v_idx == v1?v2:v1;
        
        double diff = get_delta(true_vertex);
        
        double old_value = d_deltaBuffer[true_vertex];
        if (true) {              
            swap(true_vertex, diff);
        } 
    }
    return ;
    
}

__global__ void fix_ds_wall(int width, int height, int depth)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index>=count_f_ds_wall) return;
    int i = false_ds_wall[index];
    int false_index = false_ds_wall_index[i];
    // false_index's paired face is wrong;
    int e = false_index;
    int dec_e = false_index;
    int true_face_id = paired_edges[e];
    int false_face_id = dec_paired_edges[e];
    if(true_face_id != false_face_id)
    {
        
        if(true_face_id >= 0 && false_face_id >= 0 && 
           true_face_id < num_Faces && paired_faces[true_face_id] == e 
           && false_face_id < num_Faces && dec_paired_faces[false_face_id] == dec_e 
        )
        {   
            
            int true_face = true_face_id;
            int false_face = false_face_id;
            
            int v1, v2;
            getVertexIDsFromEdgeID(dec_e, width, height, depth, v1, v2);

            int v1_1, v2_1, v3;
            getVerticesFromTriangleID(true_face, width, height, depth, v1_1, v2_1, v3);
           
            if(v1_1!=v1 && v1_1!=v2 ) v3 = v1_1;
            if(v2_1!=v1 && v2_1!=v2 ) v3 = v2_1;

            double diff = get_delta(v3);
            
            if(abs(diff)>1e-16*range)
            {
                // printf("type: %d %d %d %.17f\n", e, true_face_id, false_face_id, diff);
                double old_value = d_deltaBuffer[v3];
                if (true) {              
                    swap(v3, diff);
                } 
            }
            else
            {
                
                for(int m:{v1,v2})
                {
                    
                    diff = get_delta(m);
                    // printf("type: %.17f\n", diff);
                    double old_value = d_deltaBuffer[m];
                    if (true) {              
                        swap(m, diff);
                    }
                }
            }
            
        }
        else if(true_face_id == -1 || true_face_id >= num_Faces || (true_face_id < num_Faces && true_face_id >= 0 && paired_faces[true_face_id] != e))
        {
            // printf("type2: %d\n", index);
            int false_edge = false_index;
            int v1, v2;
            getVertexIDsFromEdgeID(false_edge, width, height, depth, v1, v2);

            int sortedV1 = !compare_vertices(v1, v2,1)? v2:v1;
            int sortedV2 = !compare_vertices(v1, v2,1)? v1:v2;
            // printf("type2: %.17f\n", decp_data[sortedV1] - input_data[sortedV1] + bound);
            // if(decp_data[sortedV1] - input_data[sortedV1] + bound > 1e-16 * range)
            // {
            //     double diff = ((input_data[sortedV1]-bound) - decp_data[sortedV1]) / 2.0;
                
            //     double old_value = d_deltaBuffer[sortedV1];
            //     if (true) {              
            //         swap(sortedV1, diff);
            //     } 
            // }
            // else
            // {
            //     double diff = ((input_data[sortedV2]-bound) - decp_data[sortedV2]) / 2.0;
                
            //     double old_value = d_deltaBuffer[sortedV2];
            //     if (true) {              
            //         swap(sortedV2, diff);
            //     } 
            // }
            for(int m:{v1,v2})
            {
                double diff = get_delta(m);
                
                double old_value = d_deltaBuffer[m];
                if (true) {              
                    swap(m, diff);
                } 
            }
        }
        else if(false_face_id == -1 || false_face_id >= num_Faces || (false_face_id < num_Faces && false_face_id >= 0 && dec_paired_faces[false_face_id] != dec_e))
        {
            // printf("type3: %d\n", index);
            int true_face = true_face_id;
            int v1, v2;
            getVertexIDsFromEdgeID(dec_e,  width, height, depth, v1, v2);
            int v1_1, v2_1, v3;
            getVerticesFromTriangleID(true_face, width, height, depth, v1_1, v2_1, v3);
            
            if(v1_1!=v1 and v1_1!=v2) v3 = v1_1;
            else if(v2_1!=v1 and v2_1!=v2) v3 = v2_1;
            double diff = get_delta(v3);
                
            double old_value = d_deltaBuffer[v3];
            if (true) {              
                swap(v3, diff);
            } 
        }
    }

}

__device__ double getlargest_v(int id, int dim, int width, int height, int depth, int type = 0)
{
    double *data = input_data;

    if (type == 1) {
        data = decp_data;
    }

    if(dim == 0)
    {
        return data[id];
    }

    if(dim == 1)
    {
        int edge = id;
        int v1, v2;
        getVertexIDsFromEdgeID(edge, width, height, depth, v1, v2);
        return compare_vertices(v1,v2,type)?data[v1]:data[v2];
    }

    if(dim==2)
    {
        int face = id;
        int v1, v2, v3;
        getVerticesFromTriangleID(face, width, height, depth, v1, v2, v3);
        double largest_value = data[v1];
        for(int v:{v2, v3})
        {
            if(data[v]>largest_value) largest_value = data[v];
        }
        return largest_value;
    }

    if(dim==3)
    {
        int tetra = id;
        int v1, v2, v3, v4;
        getVerticesFromTetrahedraID(tetra, width, height, depth, v1, v2, v3, v4);
        double largest_value = data[v1];
        for(int v:{v2, v3, v4})
        {
            if(data[v]>largest_value) largest_value = data[v];
        }
        return largest_value;
    }
}

// __device__ void reverse(SimpleStack vpath, int width, int depth, int height, int type = 0)
// {
    
//     double *data = input_data;
//     int *paired_faces1 = paired_faces;
//     int *paired_tetrahedras1 = paired_tetrahedras;
//     int *paired_edges1 = paired_edges;
//     if (type == 1) {
//         data = decp_data;
//         paired_faces1 = dec_paired_faces;
//         paired_tetrahedras1 = dec_paired_tetrahedras;
//         paired_edges1 = dec_paired_edges;
//     }
//     // printf("%d\n", vpath.getSize());
    
//    int size = vpath.getSize();
//    for(int i=0;i<size;i+2)
//     {
//         int edgeId = vpath.getElement(i);
//         int TriangleIndex = vpath.getElement(i+1);
        
//         int f = TriangleIndex;

//         int v1, v2, v3;
//         getVerticesFromTriangleID(f, width, height, depth, v1, v2, v3);
//         int v1_x = v1 % width;
//         int v1_y = (v1 / (width)) % height;
//         int v1_z = (v1 / (width * height)) % depth;

//         int v2_x = v2 % width;
//         int v2_y = (v2 / (width)) % height;
//         int v2_z = (v2 / (width * height)) % depth;

//         int v3_x = v3 % width;
//         int v3_y = (v3 / (width)) % height;
//         int v3_z = (v3 / (width * height)) % depth;
//         int e1_id = getEdgeID(v1_x, v1_y, v1_z, v2_x, v2_y, v2_z, width, height, depth);
//         int e2_id = getEdgeID(v1_x, v1_y, v1_z, v3_x, v3_y, v3_z, width, height, depth);
//         int e3_id = getEdgeID(v2_x, v2_y, v2_z, v3_x, v3_y, v3_z, width, height, depth);
//         for(int k:{e1_id, e2_id, e3_id}) {
            
//             if(k == edgeId) {
//                 paired_faces1[TriangleIndex] = k;
//                 break;
//             }
//         }

//         for(int i=0;i<14;i++)
//         {
            
//             int nx = v1_x + directions1[i][0];
//             int ny = v1_y + directions1[i][1];
//             int nz = v1_z + directions1[i][2];
//             for(int j=0;j<14;j++)
//             {
//                 int nx1 = v2_x + directions1[j][0];
//                 int ny1 = v2_y + directions1[j][1];
//                 int nz1 = v2_z + directions1[j][2];
//                 int neighbor = nx + ny * width + nz* (height * width);
                
//                 if(nx == nx1 && ny == ny1 && nz == nz1 && nx >=0 && nx < width && ny >= 0 & ny <height && nz >= 0 && nz<depth && neighbor < num_Vertices && neighbor >=0 )
//                 {
//                     int larger_id = v2;
//                     int larger_v = data[v2];
//                     if(data[v1]>data[v2] || (data[v1]==data[v2] && v1 > v2))
//                     {
//                         larger_id = v1;
//                         larger_v = data[v1];
//                     }
//                     // check if could be paired
//                     int triangleID = getTriangleID(v1, v2, neighbor, width, height, depth );
                    
//                     if(triangleID == TriangleIndex) 
//                     {
//                         paired_edges1[edgeId] = triangleID;
//                         break;
//                     }
                    
//                 }
//             }
//         }
        
//     }
//     return;
// }


// __device__ void reverse_path(int i, int tetra_id, int t, int width, int depth, int height, int type = 0, int direction = 0)
// {
    
//     double *data = input_data;
//     int *paired_faces1 = paired_faces;
//     int *paired_tetrahedras1 = paired_tetrahedras;
//     int *paired_edges1 = paired_edges;
//     int *paired_vertices1 = paired_vertices;
//     if (type == 1) {
//         data = decp_data;
//         paired_faces1 = dec_paired_faces;
//         paired_tetrahedras1 = dec_paired_tetrahedras;
//         paired_edges1 = dec_paired_edges;
//         paired_vertices1 = dec_paired_vertices;
//     }
//     // printf("%d\n", vpath.getSize());
   
    
//     // descending; 
//     if(direction == 1)
//     {
//         int current_tetra_id = tetra_id;
//         int face_id = t;
       
//         while(current_tetra_id !=-1 && face_id != -1 )
//         {
//             // if(i == 7263188) printf("current: %d %d %d %d\n", current_tetra_id, face_id, num_Faces, num_Tetrahedras);
//             if(paired_faces1[face_id]==-1)
//             {
//                 paired_tetrahedras1[current_tetra_id] = face_id;
//                 paired_faces1[face_id] = tetra_id;
//                 break;
//             }
            

//             int t1 = -1;
//             int t2 = -1;
//             get_cell_tetra(face_id, width, height, depth, &t1, &t2);

//             int tmp = current_tetra_id;
//             current_tetra_id = t1==tmp?t2:t1;
//             if(current_tetra_id==-1) face_id = -1;
//             else face_id = paired_tetrahedras1[current_tetra_id];
//             // if(i == 7263188) printf("current: %d %d %d %d %d\n", i, current_tetra_id, face_id, num_Faces, num_Tetrahedras);
//         };
//     }
//     // reverse_path(id, current_id, t, width, height, depth, type, 1);
//     else
//     {
        
//         int current_v_id = tetra_id;
//         int edge_id = t;
//         while(current_v_id !=-1 && edge_id != -1)
//         {
            
            
//             if(paired_edges1[edge_id]==-1)
//             {
//                 paired_vertices1[current_v_id] = edge_id;
//                 paired_edges1[edge_id] = current_v_id;
//                 break;
//             }
            

//             int t1, t2;
            
//             getVertexIDsFromEdgeID(edge_id, width, height, depth, t1, t2);

//             int tmp = current_v_id;
//             current_v_id = t1==tmp?t2:t1;
//             if(current_v_id != -1) edge_id = paired_vertices1[current_v_id];
//             else edge_id = -1;
            
//         };
        
//         // do
//         // {
            
//         //     old_id = current_id;
//         //     t = paired_tetrahedras1[current_id];
//         //     vpath.push(current_id);
//         //     if(t==-1)
//         //     { 
//         //         break;
//         //     }


//         //     int connectedFace = t;
//         //     vpath.push(t);
//         //     if(paired_faces1[t] == -1 || (paired_faces1[t] != -1 && paired_tetrahedras1[paired_faces1[t]] != t))
//         //     {
//         //         break;
//         //     }
//         //     int t1 = -1;
//         //     int t2 = -1;
//         //     get_cell_tetra(t, width, height, depth, &t1, &t2);
//         //     current_id = t1 == current_id?t2:t1;
//         // } while(current_id!=old_id and current_id != -1);
//         // int size = vpath.getSize();
//         // printf("%d\n", size);
//         // for(int i=size-1;i>0;i-=2)
//         // {
//         //     int TetraId = vpath.getElement(i);
//         //     int FaceId = vpath.getElement(i-1);

//         //     paired_tetrahedras1[TetraId] = FaceId;
//         //     paired_faces1[FaceId] = TetraId;
//         // }
//     }
    
//     return;
// }

// __global__ void reverse_ds_vpath(int width, int height, int depth, int type=0)
// {
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
    
//     double *data = input_data;
//     int *paired_faces1 = paired_faces;
//     int *paired_tetrahedras1 = paired_tetrahedras;
//     int *paired_edges1 = paired_edges;
//     int *paired_vertices1 = paired_vertices;
//     double persistence1 = persistence;
//     if (type == 1) {
//         data = decp_data;
//         paired_faces1 = dec_paired_faces;
//         paired_tetrahedras1 = dec_paired_tetrahedras;
//         paired_edges1 = dec_paired_edges;
//         paired_vertices1 = dec_paired_vertices;
//         persistence1 = dec_persistence;
//     }
//     if(i>=num_Edges or paired_edges1[i]!=-1) return;
//     // descending, still from edges;
//     int v1, v2;
//     getVertexIDsFromEdgeID(i, width, height, depth, v1, v2);
//     int v = v1;

//     int current_id = v1;
//     int connectedEdge = -1;
    
    
//     double birth = getlargest_v(i, 1, width, height, depth, type);
//     bool f_sign = false;
//     do
//     {
//         v = current_id;
//         if(abs(data[v] - birth) >= persistence) 
//         {   
//             f_sign = true;
//             break;
//         }

//         if(paired_vertices1[v]==-1)
//         {
//             break;
//         }

//         connectedEdge = paired_vertices1[v];
        
//         if(abs(getlargest_v(connectedEdge, 1, width, height, depth, type) - birth) >= persistence)
//         {
//             f_sign = true;
//             break;
//         }
        
//         if(paired_edges1[connectedEdge] == -1) 
//         {
//             break;
//         }

//         int v1_1, v2_1;
        
//         getVertexIDsFromEdgeID(connectedEdge, width, height, depth, v1_1, v2_1);
//         current_id = v1_1 == current_id?v2_1:v1_1;
//     } while(paired_vertices1[v]!=-1 && current_id !=-1);

//     if(!f_sign)

//     {
//         if(connectedEdge == -1) connectedEdge = i;
//         // printf("%d %d\n", v, connectedEdge);
        
//         reverse_path(i, v, connectedEdge, width, height, depth, type, 0);
//     }
    
//     v = v2;
//     current_id = v2;
//     f_sign = false;
//     connectedEdge = -1;
//     do
//     {
//         v = current_id;

//         if(abs(data[v] - birth) >= persistence) 
//         {   
//             f_sign = true;
//             break;
//         }

//         if(paired_vertices1[v]==-1)
//         {
//             break;
//         }

//         connectedEdge = paired_vertices1[v];
        
//         if(abs(getlargest_v(connectedEdge, 1, width, height, depth, type) - birth) >= persistence)
//         {
//             f_sign = true;
//             break;
//         }
        
//         if(paired_edges1[connectedEdge] == -1) 
//         {
//             break;
//         }

//         int v1_1, v2_1;
//         getVertexIDsFromEdgeID(connectedEdge, width, height, depth, v1_1, v2_1);
//         current_id = v1_1 == current_id?v2_1:v1_1;
//     } while(paired_vertices1[v]!=-1);
    
    
//     if(!f_sign)

//     {
//         if(connectedEdge == -1) connectedEdge = i;
//         reverse_path(i, v, connectedEdge, width, height, depth, type, 0);
//     }
    

//     return;
// }

// __global__ void reverse_as_vpath(int width, int height, int depth, int type =0)
// {
    
//     int id = threadIdx.x + blockIdx.x * blockDim.x;
//     if(id>=num_Faces) return;
//     double *data = input_data;
//     int *paired_faces1 = paired_faces;
//     int *paired_tetrahedras1 = paired_tetrahedras;
//     int *paired_edges1 = paired_edges;
//     double persistence1 = persistence;
//     if (type == 1) {
//         data = decp_data;
//         paired_faces1 = dec_paired_faces;
//         paired_tetrahedras1 = dec_paired_tetrahedras;
//         paired_edges1 = dec_paired_edges;
//         persistence1 = dec_persistence;
//     }
    
//     if(paired_faces1[id]!=-1) return;
//     // printf("%d\n", id);
//     int t;
//     int current_id = id;
//     int old_id;
//     bool f_sign = false;
//     int v1, v2, v3;
    
//     int num =0;
//     getVerticesFromTriangleID(id, width, height, depth, v1, v2, v3);
//     int v1_x = v1 % width;
//     int v1_y = (v1 / (width)) % height;
//     int v1_z = (v1 / (width * height)) % depth;

//     int v2_x = v2 % width;
//     int v2_y = (v2 / (width)) % height;
//     int v2_z = (v2 / (width * height)) % depth;

//     int v3_x = v3 % width;
//     int v3_y = (v3 / (width)) % height;
//     int v3_z = (v3 / (width * height)) % depth;

//     double birth = getlargest_v(id, 2, width, height, depth, type);
    
//     for (int i = 0; i < 14; i++) 
//     {
//         int nx = v1_x + directions1[i][0];
//         int ny = v1_y + directions1[i][1];
//         int nz = v1_z + directions1[i][2];
//         int neighbor_id1 = nx + ny * width + nz * width * height;
        
//         if(nx >= width || ny >= height || nz >= depth || nx < 0 ||
//            ny < 0 || nz < 0 || neighbor_id1 >= num_Vertices || neighbor_id1 < 0 ) continue;

//         for (int j = 0; j < 14; j++)
//         {
            
//             int nx1 = v2_x + directions1[j][0];
//             int ny1 = v2_y + directions1[j][1];
//             int nz1 = v2_z + directions1[j][2];
//             int neighbor_id2 = nx1 + ny1 * width + nz1 * width * height;
//             if(nx1 >= width || ny1 >= height || nz1 >= depth || nx1 < 0 ||
//                 ny1 < 0 || nz1 < 0 || neighbor_id2 >= num_Vertices || neighbor_id2 < 0 ) continue;
//             for (int k = 0; k < 14; k++) {

//                 int nx2 = v3_x + directions1[k][0];
//                 int ny2 = v3_y + directions1[k][1];
//                 int nz2 = v3_z + directions1[k][2];
//                 int neighbor_id3 = nx2 + ny2 * width + nz2 * width * height;
//                 if(nx2 >= width || ny2 >= height || nz2 >= depth || nx2 < 0 ||
//                     ny2 < 0 || nz2 < 0 || neighbor_id3 >= num_Vertices || neighbor_id3 < 0 ) continue;
                
//                 if(!(neighbor_id1 == neighbor_id2 && neighbor_id2 == neighbor_id3 && neighbor_id1 == neighbor_id3 )) continue;
//                 if (neighbor_id1 == v1 || neighbor_id1 == v2 || neighbor_id1== v3) continue;

//                 int v4 = neighbor_id1;
                
//                 int tetra_id = getTetrahedraID(v1, v2, v3, v4, width, height, depth);
//                 // printf("%d %d\n", tetra_id, id);

//                 int t;
//                 int current_id = tetra_id;
//                 int old_id;
//                 bool f_sign = false;
//                 t = tetra_id;
//                 current_id = tetra_id;
//                 int connectedFace = -1;
//                 do
//                 {
                    
//                     old_id = current_id;
//                     t = paired_tetrahedras1[current_id];
//                     if(abs(getlargest_v(current_id, 3, width, height, depth, type) - birth) >= persistence1)
//                     {
//                         f_sign = true;
//                         break;
//                     }
//                     if(t==-1 || t>= num_Faces || (t>= 0 && t<num_Faces && paired_faces1[t] != current_id))
//                     { 
//                         break;
//                     }

//                     if(abs(getlargest_v(t, 2, width, height, depth, type) - birth) >= persistence1)
//                     {
//                         f_sign = true;
//                         break;
//                     }

//                     connectedFace = t;
                    
//                     if(paired_faces1[t] == -1 || (paired_faces1[t] >=0 && paired_faces1[t] < num_Tetrahedras && paired_tetrahedras1[paired_faces1[t]] != t))
//                     {
//                         break;
//                     }
//                     int t1 = -1;
//                     int t2 = -1;
//                     get_cell_tetra(t, width, height, depth, &t1, &t2);
//                     current_id = t1 == current_id?t2:t1;
//                 } while(current_id!=old_id and current_id != -1);
                
//                 if(!f_sign)
//                 {
//                     // printf("current: %d %d %d\n", current_id, connectedFace);
//                     if(connectedFace==-1)
//                     {
//                         connectedFace = id;
//                     }
//                     reverse_path(id, current_id, connectedFace, width, height, depth, type, 1);
//                 }
//                 // printf("%d %d\n", tetra_id, id);
//             }
//         }
//     }

// }

// first identify the reversed saddle-saddle connector, 
// __global__ void reverse_asceding_through_wall(int width, int height, int depth, int type=0)
// {
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     double *data = input_data;
//     int *paired_faces1 = paired_faces;
//     int *paired_tetrahedras1 = paired_tetrahedras;
//     int *paired_edges1 = paired_edges;
//     if (type == 1) {
//         data = decp_data;
//         paired_faces1 = dec_paired_faces;
//         paired_tetrahedras1 = dec_paired_tetrahedras;
//         paired_edges1 = dec_paired_edges;
//     }

//     if(i>=num_Faces or paired_faces1[i]!=-1) return;
//     int f = i;
//     int triangleId = i;
//     SimpleStack bfs(10);
//     SimpleStack saddles1(10);
//     bfs.push(triangleId);
    
//     int false_index;
//     double birth = getlargest_v(i, 2, width, height, depth, type);
//     // if(i==0)printf("over\n");
//     while(!bfs.isEmpty())
//     {
//         triangleId = bfs.pop();
        
//         if(visited_faces[triangleId]== -1)
//         {
            
//             visited_faces[triangleId] = 1;
//             int v1, v2, v3;
//             getVerticesFromTriangleID(triangleId, width, height, depth, v1, v2, v3);
//             int v1_x = v1 % width;
//             int v1_y = (v1 / (width)) % height;
//             int v1_z = (v1 / (width * height)) % depth;

//             int v2_x = v2 % width;
//             int v2_y = (v2 / (width)) % height;
//             int v2_z = (v2 / (width * height)) % depth;

//             int v3_x = v3 % width;
//             int v3_y = (v3 / (width)) % height;
//             int v3_z = (v3 / (width * height)) % depth;
//             int e1_id = getEdgeID(v1_x, v1_y, v1_z, v2_x, v2_y, v2_z, width, height, depth);
//             int e2_id = getEdgeID(v1_x, v1_y, v1_z, v3_x, v3_y, v3_z, width, height, depth);
//             int e3_id = getEdgeID(v2_x, v2_y, v2_z, v3_x, v3_y, v3_z, width, height, depth);
//             for(int j:{e1_id, e2_id, e3_id})
//             {   
//                 // if(j == -1 or j>num_Edges) printf("%d\n", triangleId);
//                 if(paired_edges[j]==-1)
//                 {
//                     if(abs(getlargest_v(j, 1, width, height, depth, type) - birth) < persistence)
//                     {
//                         saddles1.push(j);
//                     }
//                     continue;
//                 }
//                 if(paired_edges[j]!=dec_paired_edges[j])
//                 {
//                     int idx_fp_saddle = atomicAdd(&count_f_ds_wall, 1);// in one instruction
//                     false_ds_wall[idx_fp_saddle] = i;
//                     false_ds_wall_index[i] = j;
//                     return;
//                 }
//                 else
//                 {
//                     int pairedCellId = paired_edges[j];
//                     if(pairedCellId != triangleId and pairedCellId!=-1 and paired_faces[pairedCellId] == j) bfs.push(pairedCellId);
//                 }
//             }
//         }
//     }
//     bfs.clear1();
//     // printf("over\n");
    
//     while(!saddles1.isEmpty())
//     {
//         SimpleStack vpath(10);
//         int j = saddles1.pop();
//         int currentID = -1;
//         vpath.push(j);
//         bool found = false;
//         int v1, v2;
    
//         getVertexIDsFromEdgeID(j,width, height, depth , v1, v2);
//         int v1_x = v1 % width;
//         int v1_y = (v1 / (width)) % height;
//         int v1_z = (v1 / (width * height)) % depth;

//         int v2_x = v2 % width;
//         int v2_y = (v2 / (width)) % height;
//         int v2_z = (v2 / (width * height)) % depth;
//         for(int i=0;i<14;i++)
//         {
            
//             int nx = v1_x + directions1[i][0];
//             int ny = v1_y + directions1[i][1];
//             int nz = v1_z + directions1[i][2];
//             for(int j=0;j<14;j++)
//             {
//                 int nx1 = v2_x + directions1[j][0];
//                 int ny1 = v2_y + directions1[j][1];
//                 int nz1 = v2_z + directions1[j][2];
//                 int neighbor = nx + ny * width + nz* (height * width);
                
//                 if(nx == nx1 && ny == ny1 && nz == nz1 && nx >=0 && nx < width && ny >= 0 & ny <height && nz >= 0 && nz<depth && neighbor < num_Vertices && neighbor >=0 )
//                 {
//                     int larger_id = v2;
//                     int larger_v = data[v2];
//                     if(data[v1]>data[v2] || (data[v1]==data[v2] && v1 > v2))
//                     {
//                         larger_id = v1;
//                         larger_v = data[v1];
//                     }
//                     // check if could be paired
//                     int triangleID = getTriangleID(v1, v2, neighbor, width, height, depth );
//                     if(found) break;
//                     if(visited_faces[triangleID]==1) {
//                         // saddle1 can be adjacent to saddle2 on the wall
//                         if(paired_faces1[triangleID]==-1) {
//                             found = true;
//                             vpath.push(triangleID);
//                             break;
//                         }
//                         currentID = triangleID;
//                     }
                    
//                 }
//             }
//         }
        

//         int oldId;
//         if(!found)
//         {
//             do 
//             {

//                 oldId = currentID;
//                 vpath.push(currentID);
//                 if(paired_faces1[currentID]==-1 || paired_faces1[currentID]>=num_Edges || (paired_faces1[currentID] >= 0 && paired_faces1[currentID] < num_Edges && paired_edges1[paired_faces1[currentID]] != currentID)) {
//                     break;
//                 }
//                 const int connectedEdgeId
//                     = paired_faces1[currentID];
//                 vpath.push(connectedEdgeId);
//                 if(paired_edges1[connectedEdgeId] == -1) {
//                     break;
//                 }
//                 int v1, v2;
    
//                 getVertexIDsFromEdgeID(connectedEdgeId,width, height, depth , v1, v2);
//                 int v1_x = v1 % width;
//                 int v1_y = (v1 / (width)) % height;
//                 int v1_z = (v1 / (width * height)) % depth;

//                 int v2_x = v2 % width;
//                 int v2_y = (v2 / (width)) % height;
//                 int v2_z = (v2 / (width * height)) % depth;
//                 for(int i=0;i<14;i++)
//                 {
                    
//                     int nx = v1_x + directions1[i][0];
//                     int ny = v1_y + directions1[i][1];
//                     int nz = v1_z + directions1[i][2];
//                     for(int j=0;j<14;j++)
//                     {
//                         int nx1 = v2_x + directions1[j][0];
//                         int ny1 = v2_y + directions1[j][1];
//                         int nz1 = v2_z + directions1[j][2];
//                         int neighbor = nx + ny * width + nz* (height * width);
                        
//                         if(nx == nx1 && ny == ny1 && nz == nz1 && nx >=0 && nx < width && ny >= 0 & ny <height && nz >= 0 && nz<depth && neighbor < num_Vertices && neighbor >=0 )
//                         {
//                             int larger_id = v2;
//                             int larger_v = data[v2];
//                             if(data[v1]>data[v2] || (data[v1]==data[v2] && v1 > v2))
//                             {
//                                 larger_id = v1;
//                                 larger_v = data[v1];
//                             }
//                             // check if could be paired
//                             int triangleID = getTriangleID(v1, v2, neighbor, width, height, depth );
//                             if(visited_faces[triangleID] == 1 and triangleID != oldId) 
//                             {
//                                 currentID = triangleID;
//                             }
                            
//                         }
//                     }
//                 }

//                 // stop at convergence caused by boundary effect
//             } while(currentID != oldId and currentID!=-1);
//         }
//         // reversing process
//         reverse(vpath, width, height, depth, type);
//         vpath.clear1();
//     }
//     saddles1.clear1();
//     // printf("reversed\n");
// }

// __global__ void get1saddlemin(int type = 0)
// {
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
    
//     double *data = input_data;
//     int *paired_faces1 = paired_faces;
//     int *paired_tetrahedras1 = paired_tetrahedras;
//     int *paired_edges1 = paired_edges;
//     if (type == 1) {
//         data = decp_data;
//         paired_faces1 = dec_paired_faces;
//         paired_tetrahedras1 = dec_paired_tetrahedras;
//         paired_edges1 = dec_paired_edges;
//     }
//     if(i>=num_Edges or paired_edges1[i]!=-1) return;
//     int v1_1, v2_1;
//     getVertexIDsFromEdgeID(i, width, height, depth, v1_1, v2_1);
//     int v = v1_1;
//     int current_id = v1_1;
    
//     double birth = getlargest_v(i, 1, type);
//     int connectedEdge;
    

//     bool f_sign = false;
//     do
//     {
        
//         v = current_id;
//         if(paired_vertices1[v]==-1)
//         {
//             break;
//         }

//         connectedEdge = paired_vertices1[v];
        
    

//         if(paired_edges1[connectedEdge]==-1) 
//         {
//             break;
//         }

//         int v1, v2;
//         getVertexIDsFromEdgeID(connectedEdge, width, height, depth, v1, v2);

//         current_id = v1 == current_id?v2:v1;
//         v = current_id;
//     } while(paired_vertices1[v]!=-1);

//     int v_id = v;
//     if(abs(v->value - birth) > persistence1 && v->filtered)
//     {
//        filtered_vertices1[v] = false;
//        filtered_vertices1[v] = false;
//     }

//     v = v2_1;
//     current_id = v2_1;

//     do
//     {
        
//         v = current_id;
//         if(paired_vertices1[v]==-1)
//         {
//             break;
//         }

//         connectedEdge = paired_vertices1[v];
        
    

//         if(paired_edges1[connectedEdge]==-1) 
//         {
//             break;
//         }

//         int v1, v2;
//         getVertexIDsFromEdgeID(connectedEdge, width, height, depth, v1, v2);

//         current_id = v1 == current_id?v2:v1;
//         v = current_id;
//     } while(paired_vertices1[v]!=-1);
    
//     v_id = v->id;
//     edges1[i].v_persistence[1] = v_id;
//     if(abs(v->value - birth) > persistence1 && v->filtered)
//     {
//         v->filtered = false;
//     }
//     return;
// }

// __global__ void get_as_integral_line(int width, int height, int depth)
// {
//     // start from each regular vertices
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index >= numVertices) return;

//     double *data1 = input_data;
//     int *paired_vertices1 = paired_vertices;
//     int *paired_edges1 = paired_edges;
//     if(type==1)
//     {
//         data1 = decp_data;
//         paired_vertices1 = dec_paired_vertices;
//         paired_edges1 = dec_paired_edges;
//     }
    
// }

__global__ void init_counter()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_Vertices) return;

    delta_counter[index] = 1;
}

__global__ void init_neighbors()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_Vertices) return;
    for(int i=0;i<14;i++)
    {
        neighbors_index[14*index+i] = -1;
    }
    
}

int main(int argc, char** argv) {
    std::cout.precision(std::numeric_limits<double>::max_digits10);
    // Initialize the grid with some values
    
    std::cout.precision(std::numeric_limits<double>::max_digits10);
    std::string dimension = argv[1];
    range1 = std::stod(argv[2]);
    std::string compressor_id = argv[3];

    preserve_saddles = std::stoi(argv[4]);
    preserve_vpath = std::stoi(argv[5]);
    
    preserve_geometry = std::stoi(argv[6]);
    filtration = std::stoi(argv[7]);
    preserve_connectors = std::stoi(argv[8]);
    preserve_type = std::stoi(argv[9]);
    edit_type1 = std::stoi(argv[10]);

    std::istringstream iss(dimension);
    char delimiter;
    std::string filename;
    if (std::getline(iss, filename, ',')) {
        if (iss >> width >> delimiter && delimiter == ',' &&
            iss >> height >> delimiter && delimiter == ',' &&
            iss >> depth) {
            std::cout << "Filename: " << filename << std::endl;
            std::cout << "Width: " << width << std::endl;
            std::cout << "Height: " << height << std::endl;
            std::cout << "Depth: " << depth << std::endl;
        } else {
            std::cerr << "Parsing error for dimensions" << std::endl;
        }
    } else {
        std::cerr << "Parsing error for filename" << std::endl;
    }
    inputfilename = filename+".bin";
    size2 = width * height * depth;

    std::vector<double> input_data1;
    std::vector<double> decp_data1;
    std::string command;
    int result;
    
    

    input_data1 = getdata2(inputfilename);
    auto min_it = std::min_element(input_data1.begin(), input_data1.end());
    auto max_it = std::max_element(input_data1.begin(), input_data1.end());
    minValue = *min_it;
    maxValue = *max_it;


    bound1 = (maxValue-minValue)*range1;

    double d = (maxValue-minValue);
    cudaMemcpyToSymbol(bound, &bound1, sizeof(double), 0, cudaMemcpyHostToDevice);
    
    checkCudaError(cudaMemcpyToSymbol(persistence, &bound1, sizeof(double), 0, cudaMemcpyHostToDevice), "persistence1");
    checkCudaError(cudaMemcpyToSymbol(preserve_type1, &preserve_type, sizeof(int), 0, cudaMemcpyHostToDevice), "persistence1");
    
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(6) << range1;


    std::ostringstream oss1;
    oss1 << std::scientific << std::setprecision(17) << bound1;

    auto startt = std::chrono::high_resolution_clock::now();
    cudaMemcpyToSymbol(range, &d, sizeof(double), 0, cudaMemcpyHostToDevice);
    
    cout<<"bound1: "<<bound1<<endl;

    std::string decp_filename = "/pscratch/sd/y/yuxiaoli/MSCz/decompressed_data/decp_"+filename+"_"+oss.str()+compressor_id+"_.bin";
    cpfilename = "/pscratch/sd/y/yuxiaoli/MSCz/compressed_data/compressed_"+filename+"_"+oss.str()+".sz";
    std::string fix_path = "/pscratch/sd/y/yuxiaoli/MSCz/fixed_decp_data/fixed_decp_"+filename+"_"+oss.str()+".bin";
    cout<<"decp_filename: "<<decp_filename<<"fixed_decp_file_name:"<<fix_path<<endl;
    if(compressor_id=="sz3"){
        
        command = "sz3 -i " + inputfilename + " -z " + cpfilename +" -o "+ decp_filename + " -d " + " -1 " + std::to_string(size2) + " -M "+"REL "+oss.str()+" -a";
        std::cout << "Executing command: " << command << std::endl;
        result = std::system(command.c_str());
        if (result == 0) {
            std::cout << "Compression successful." << std::endl;
        } else {
            std::cout << "Compression failed." << std::endl;
        }
    }
    else if(compressor_id=="zfp"){
        cpfilename = "/pscratch/sd/y/yuxiaoli/MSCz/compressed_data/compressed_"+filename+"_"+oss.str()+".zfp";
        // zfp -i ~/msz/experiment_data/finger.bin -z compressed.zfp -d -r 0.001
        // decp_filename = "/pscratch/sd/y/yuxiaoli/MSCz/decompressed_data/decp_"+filename+"_"+std::to_string(bound1)+compressor_id+"_.bin";
        
        command = "zfp -i " + inputfilename + " -z " + cpfilename +" -o "+decp_filename + " -d " + " -1 " + std::to_string(size2)+" -a "+oss1.str()+" -s";
        std::cout << "Executing command: " << command << std::endl;
       
        result = std::system(command.c_str());
        if (result == 0) {
            std::cout << "Compression successful." << std::endl;
        } else {
            std::cout << "Compression failed." << std::endl;
        }
    }
    cout<<"decp_filename: "<<decp_filename<<"fixed_decp_file_name:"<<fix_path<<endl;
    auto end = std::chrono::high_resolution_clock::now();
    // return 0;
    std::chrono::duration<double> duration = end - startt;

    compression_time = duration.count();

    decp_data1 = getdata2(decp_filename);
    // cout<<decp_data1[166423]<<","<<input_data1[166423]-bound-decp_data1[166423]<<endl;
    // return 0;
    min_it = std::min_element(decp_data1.begin(), decp_data1.end());
    max_it = std::max_element(decp_data1.begin(), decp_data1.end());
    minValue = *min_it;
    maxValue = *max_it;
    
    double persistence2 = range1 * (maxValue - minValue);
    
    

    checkCudaError(cudaMemcpyToSymbol(dec_persistence, &persistence2, sizeof(double), 0, cudaMemcpyHostToDevice), "error persistence");
    
    

    std::vector<double> decp_data_copy(decp_data1);
    
    // translate data to gpu;

    startt = std::chrono::high_resolution_clock::now();
    double* input_temp;
    checkCudaError(cudaMalloc(&input_temp, size2  * sizeof(double)), "error 1");
    cudaMemcpy(input_temp, input_data1.data(), size2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(input_data, &input_temp, sizeof(double*));

    double* decp_temp;
    checkCudaError(cudaMalloc(&decp_temp, size2  * sizeof(double)), "error 2");
    checkCudaError(cudaMemcpy(decp_temp, decp_data1.data(), size2 * sizeof(double), cudaMemcpyHostToDevice), "error 3");
    cudaMemcpyToSymbol(decp_data, &decp_temp, sizeof(double*));



    

    
    // calculate numofv, numof e...
    int numVertices = size2;
    
    int numEdges = (width - 1) * height * depth
                   + width * (height - 1) * depth
                   + width * height * (depth - 1)
                   + (width - 1) * (height - 1) * depth
                   + (width - 1) * height * (depth - 1)
                   + width * (height - 1) * (depth - 1)
                   + (width - 1) * (height - 1) * (depth-1);
    
    int numFaces = 2* ((width - 1) * (height - 1) * depth
                   + width * (height - 1) * (depth - 1)
                   + (width - 1) * height * (depth - 1)) + 6 * (width-1)*(height-1)*(depth-1);
                   
    int numTetrahedras = 6 * (width - 1) * (height - 1) * (depth - 1);
    

    // initilization of timings;
    float elapsedTime = 0.0;
    float build_cells = 0.0;
    float gradient_pairing = 0.0;
    float get_f_cp = 0.0;
    float get_f_path = 0.0;
    float fix_f_cp = 0.0;
    float fix_f_vpath = 0.0;
    float init_cells = 0.0;

    
    
    float gradient_pairing_sub = 0.0;
    float get_f_cp_sub = 0.0;
    float get_f_path_sub = 0.0;
    float fix_f_cp_sub = 0.0;
    float fix_f_vpath_sub = 0.0;
    float init_cells_sub = 0.0;


    // create time counter;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    

    // allocate memory for cells;
    // calculate the timing;
    cudaEventRecord(start, 0);

    // translate number to gpu;
    cudaMemcpyToSymbol(num_Vertices, &numVertices, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(num_Edges, &numEdges, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(num_Faces, &numFaces, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(num_Tetrahedras, &numTetrahedras, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(edit_type, &edit_type1, sizeof(int), 0, cudaMemcpyHostToDevice);
    checkCudaError(cudaMemcpyToSymbol(simplification, &simplification1, sizeof(int), 0, cudaMemcpyHostToDevice), "error persistence");
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    build_cells+=elapsedTime;

    
    
    // initilization of d_deltabuffer
    double* buffer_temp;
    cudaMalloc(&buffer_temp, size2  * sizeof(double));
    cudaMemcpyToSymbol(d_deltaBuffer, &buffer_temp, sizeof(double*));
    // copy and allocate data to device. original data.

    // copy and allocate data to device. decp data.
    
    // allocate space for dec_cells
    
    int *vertices_temp;  
    cudaMalloc(&vertices_temp, numVertices * sizeof(int));
    cudaMemset(vertices_temp, -1, numVertices * sizeof(int));
    cudaMemcpyToSymbol(paired_vertices, &vertices_temp, sizeof(int*));

    int *edges_temp;  
    cudaMalloc(&edges_temp, numEdges * sizeof(int));
    cudaMemset(edges_temp, -1, numEdges  * sizeof(int));
    cudaMemcpyToSymbol(paired_edges, &edges_temp, sizeof(int*));

    int *faces_temp;  
    cudaMalloc(&faces_temp, numFaces * sizeof(int));
    cudaMemset(faces_temp, -1, numFaces  * sizeof(int));
    cudaMemcpyToSymbol(paired_faces, &faces_temp, sizeof(int*));

    int *visited_mask;  
    cudaMalloc(&visited_mask, numFaces * sizeof(int));
    cudaMemset(visited_mask, -1, numFaces  * sizeof(int));
    cudaMemcpyToSymbol(visited_faces, &visited_mask, sizeof(int*));

    int *tetra_temp;  
    cudaMalloc(&tetra_temp, numTetrahedras * sizeof(int));
    cudaMemset(tetra_temp, -1, numTetrahedras  * sizeof(int));
    cudaMemcpyToSymbol(paired_tetrahedras, &tetra_temp, sizeof(int*));

    int *dec_vertices_temp;  
    cudaMalloc(&dec_vertices_temp, numVertices * sizeof(int));
    cudaMemset(dec_vertices_temp, -1, numVertices * sizeof(int));
    cudaMemcpyToSymbol(dec_paired_vertices, &dec_vertices_temp, sizeof(int*));

    int *dec_edges_temp;  
    cudaMalloc(&dec_edges_temp, numEdges * sizeof(int));
    cudaMemset(dec_edges_temp, -1, numEdges  * sizeof(int));
    cudaMemcpyToSymbol(dec_paired_edges, &dec_edges_temp, sizeof(int*));

    int *dec_faces_temp;  
    cudaMalloc(&dec_faces_temp, numFaces * sizeof(int));
    cudaMemset(dec_faces_temp, -1, numFaces  * sizeof(int));
    cudaMemcpyToSymbol(dec_paired_faces, &dec_faces_temp, sizeof(int*));

    int *dec_tetra_temp;  
    cudaMalloc(&dec_tetra_temp, numTetrahedras * sizeof(int));
    cudaMemset(dec_tetra_temp, -1, numTetrahedras  * sizeof(int));
    cudaMemcpyToSymbol(dec_paired_tetrahedras, &dec_tetra_temp, sizeof(int*));
    
    cudaDeviceSynchronize();
    
    dim3 blockSize(256);
    
    int blockSize1 = 256;
    dim3 gridSize((numVertices + blockSize.x - 1) / blockSize.x);
    dim3 gridSizeEdge((numEdges + blockSize.x - 1) / blockSize.x);
    dim3 gridSizeFace((numFaces + blockSize.x - 1) / blockSize.x);
    dim3 gridSizeTetrahedra((numTetrahedras + blockSize.x - 1) / blockSize.x);


    
    int *neighbor_temp;  
    cudaMalloc(&neighbor_temp, 14 * numVertices * sizeof(int));
    cudaMemcpyToSymbol(neighbors_index, &neighbor_temp, sizeof(int*));
    cudaDeviceSynchronize();
    
    init_neighbors<<<gridSize, blockSize>>>();
    cudaDeviceSynchronize();

    cudaEventRecord(start, 0);
    init_vertices<<<gridSize, blockSize>>>(1);
    cudaDeviceSynchronize();
    init_edges<<<gridSizeEdge, blockSize>>>(1);
    cudaDeviceSynchronize();
    init_faces<<<gridSizeFace, blockSize>>>(1);
    cudaDeviceSynchronize();
    init_tetras<<<gridSizeTetrahedra, blockSize>>>(1);
    cudaDeviceSynchronize();


    processVertices<<<gridSize, blockSize>>>(numVertices, width, height, depth);
    cudaDeviceSynchronize();

    edgePairingKernel<<<gridSizeEdge, blockSize>>>(width, height, depth, numVertices, numEdges, numFaces);
    cudaDeviceSynchronize();
    
    trianglePairingKernel<<<gridSizeFace, blockSize>>>(width, height, depth, numVertices, numEdges, numFaces, numTetrahedras);
    cudaDeviceSynchronize();


    
    edgePairingKernel<<<gridSizeEdge, blockSize>>>(width, height, depth, numVertices, numEdges, numFaces, 0, 1);
    cudaDeviceSynchronize();
    trianglePairingKernel<<<gridSizeFace, blockSize>>>(width, height, depth, numVertices, numEdges, numFaces, numTetrahedras, 0, 1);
    cudaDeviceSynchronize();
    
    if(preserve_vpath==1 && preserve_geometry == 0)
    {
        int *connected_max_saddle1;  
        cudaMalloc(&connected_max_saddle1, 2*numFaces * sizeof(int));
        checkCudaError(cudaMemset(connected_max_saddle1, -1, 2*numFaces * sizeof(int)), "error ajl");
        cudaMemcpyToSymbol(connected_max_saddle, &connected_max_saddle1, sizeof(int*));

        int *connected_min_saddle1;  
        cudaMalloc(&connected_min_saddle1, 2*numEdges * sizeof(int));
        checkCudaError(cudaMemset(connected_min_saddle1, -1, 2*numEdges  * sizeof(int)), "error ahk");
        cudaMemcpyToSymbol(connected_min_saddle, &connected_min_saddle1, sizeof(int*));
        cudaDeviceSynchronize();

        get_orginal_saddle_min_pairs<<<gridSizeEdge, blockSize>>>(width, height, depth);
        get_orginal_saddle_max_pairs<<<gridSizeFace, blockSize>>>(width, height, depth);
        cudaDeviceSynchronize();
    }
    
    // if(filtration == 1)
    // {
        
    //     reverse_as_vpath<<<gridSizeFace, blockSize>>>(width, height, depth);
    //     cudaDeviceSynchronize();
    //     // return 0;
    //     cout<<"simplification_end"<<endl;
    //     reverse_ds_vpath<<<gridSizeEdge, blockSize>>>(width, height, depth);
    //     cudaDeviceSynchronize();
    // }

    int *u_temp;  
    cudaMalloc(&u_temp, numVertices * sizeof(int));
    // cudaMemset(u_temp, -1, numVertices  * sizeof(int));
    cudaMemcpyToSymbol(unchangeable_vertices, &u_temp, sizeof(int*));
    cudaDeviceSynchronize();

    int *counter_temp;  
    cudaMalloc(&counter_temp, numVertices * sizeof(int));
    // cudaMemset(u_temp, -1, numVertices  * sizeof(int));
    cudaMemcpyToSymbol(delta_counter, &counter_temp, sizeof(int*));
    cudaDeviceSynchronize();

    
    if(filtration == 1)
    {
        filter_vertices<<<gridSize, blockSize>>>(width, height, depth);
        cudaDeviceSynchronize();
        filter_edges<<<gridSizeEdge, blockSize>>>(width, height, depth);
        cudaDeviceSynchronize();
        filter_faces<<<gridSizeFace, blockSize>>>(width, height, depth);
        cudaDeviceSynchronize();
        filter_tetrahedras<<<gridSizeTetrahedra, blockSize>>>(width, height, depth);
        cudaDeviceSynchronize();
    }
    
    // return 0;
    // reverse_asceding_through_wall<<<gridSizeFace, blockSize>>>(width, height, depth);
    // cudaDeviceSynchronize();
    
    processVertices<<<gridSize, blockSize>>>(numVertices, width, height, depth, 1);
    cudaDeviceSynchronize();
    
    edgePairingKernel<<<gridSizeEdge, blockSize>>>(width, height, depth, numVertices, numEdges, numFaces, 1);
    cudaDeviceSynchronize();
    
    trianglePairingKernel<<<gridSizeFace, blockSize>>>(width, height, depth, numVertices, numEdges, numFaces, numTetrahedras,1);
    cudaDeviceSynchronize();
    

    
    edgePairingKernel<<<gridSizeEdge, blockSize>>>(width, height, depth, numVertices, numEdges, numFaces, 1 ,1);
    cudaDeviceSynchronize();
    trianglePairingKernel<<<gridSizeFace, blockSize>>>(width, height, depth, numVertices, numEdges, numFaces, numTetrahedras,1, 1);
    cudaDeviceSynchronize();
    
    cout<<"simplification"<<endl;
    // if(filtration == 1)
    // {
    //     reverse_as_vpath<<<gridSizeFace, blockSize>>>(width, height, depth,1);
    //     cudaDeviceSynchronize();
        
    //     reverse_ds_vpath<<<gridSizeEdge, blockSize>>>(width, height, depth,1);
    //     cudaDeviceSynchronize();
    // }
    cout<<"simplification_end"<<endl;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    gradient_pairing+=elapsedTime;
    
    // initialization for false counts;
    int initialValue = 0;
    cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(int));
    cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(int));
    cudaMemcpyToSymbol(count_f_saddle, &initialValue, sizeof(int));
    cudaMemcpyToSymbol(count_f_2_saddle, &initialValue, sizeof(int));
    // initialization for false_arrays;
    int *min_temp;  
    cudaMalloc(&min_temp, numVertices * sizeof(int));
    cudaMemcpyToSymbol(false_min, &min_temp, sizeof(int*));

    


    int *saddle_temp;  
    cudaMalloc(&saddle_temp, numEdges * sizeof(int));
    cudaMemcpyToSymbol(false_saddle, &saddle_temp, sizeof(int*));

    
    int *saddle_2temp;  
    cudaMalloc(&saddle_2temp, numFaces * sizeof(int));
    cudaMemcpyToSymbol(false_2saddle, &saddle_2temp, sizeof(int*));

    int *max_temp;  
    cudaMalloc(&max_temp, numFaces * sizeof(int));
    cudaMemcpyToSymbol(false_max, &max_temp, sizeof(int*));

    int *vpath_ds_temp;  
    cudaMalloc(&vpath_ds_temp, numEdges * sizeof(int));
    cudaMemcpyToSymbol(false_ds_vpath, &vpath_ds_temp, sizeof(int*));

    int *vpath_ds_temp_index;  
    cudaMalloc(&vpath_ds_temp_index, 2 * numEdges * sizeof(int));
    cudaMemcpyToSymbol(false_ds_vpath_index, &vpath_ds_temp_index, sizeof(int*));

    int *vpath_temp;  
    cudaMalloc(&vpath_temp, numFaces * sizeof(int));
    cudaMemcpyToSymbol(false_as_vpath, &vpath_temp, sizeof(int*));

    int *vpath_temp_index;  
    cudaMalloc(&vpath_temp_index, 2 * numFaces * sizeof(int));
    cudaMemcpyToSymbol(false_as_vpath_index, &vpath_temp_index, sizeof(int*));

    

    int *wpath_temp;  
    cudaMalloc(&wpath_temp, numFaces * sizeof(int));
    cudaMemcpyToSymbol(false_ds_wall, &wpath_temp, sizeof(int*));

    int *wpath_temp_index;  
    cudaMalloc(&wpath_temp_index, numFaces * sizeof(int));
    cudaMemcpyToSymbol(false_ds_wall_index, &wpath_temp, sizeof(int*));

    int *wa_path_temp;  
    cudaMalloc(&wa_path_temp, numEdges * sizeof(int));
    cudaMemcpyToSymbol(false_as_wall, &wa_path_temp, sizeof(int*));

    int *wa_path_temp_index;  
    cudaMalloc(&wa_path_temp_index, numEdges * sizeof(int));
    cudaMemcpyToSymbol(false_as_wall_index, &wa_path_temp_index, sizeof(int*));

    
    // get_simplified_minimum
    // get1saddlemin<<<gridSizeEdge, blockSize>>>();
    // cudaDeviceSynchronize();
    // get1saddlemin<<<gridSizeEdge, blockSize>>>(1);
    // cudaDeviceSynchronize();

    // get_simplified_maximum
    // get2saddlemax_connector<<<gridSizeFace, blockSize>>>();
    // cudaDeviceSynchronize();
    // get2saddlemax_connector<<<gridSizeFace, blockSize>>>(1);
    // cudaDeviceSynchronize();
    
    // // get_false_simplified_saddles;
    // get_false_simplified_minimum<<<gridSize, blockSize>>>();
    // cudaDeviceSynchronize();
    // get_false_simplified_maximum<<<gridSizeTetrahedra, blockSize>>>();
    // cudaDeviceSynchronize();
    // get_false_simplified_1saddle<<<gridSizeEdge, blockSize>>>();
    // cudaDeviceSynchronize();
    // get_false_simplified_2saddle<<<gridSizeFace, blockSize>>>(width, height, depth);
    // cudaDeviceSynchronize();

    cudaEventRecord(start, 0);
    get_false_minimum<<<gridSize, blockSize>>>();
    get_false_saddle<<<gridSizeEdge, blockSize>>>();
    get_false_2_saddle<<<gridSizeFace, blockSize>>>();
    get_false_maximum<<<gridSizeTetrahedra, blockSize>>>();
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    get_f_cp+=elapsedTime;
    

    int host_count_f_max;
    cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
    
    int host_count_f_min;
    cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
    
    int host_count_f_saddle;
    cudaMemcpyFromSymbol(&host_count_f_saddle, count_f_saddle, sizeof(int), 0, cudaMemcpyDeviceToHost);

    int host_count_f_2_saddle;
    cudaMemcpyFromSymbol(&host_count_f_2_saddle, count_f_2_saddle, sizeof(int), 0, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    cout<<host_count_f_max<<", "<< host_count_f_min<<", "<<host_count_f_saddle<<","<<host_count_f_2_saddle<<endl;
    
    int host_count_f_as_vpath = 1;
    int host_count_f_ds_vpath = 1;
    int host_count_f_as_wall = 1;
    int host_count_f_ds_wall = 1;
    
    size_t shared_memory_size = (blockSize1 + 1) * sizeof(int);
    // cudaMemcpyToSymbol(count_f_vpath, &initialValue, sizeof(int));
    // get_false_vpath<<<blocksPerGrid_edge, threadsPerBlock>>>();
    std::vector<std::vector<float>> time_counter;
    int total_ite = 0;
    // return 0;
    int* global_memory;
    int global_capacity = 1024; // Initial global capacity, can be expanded in the device code

    cudaMalloc(&global_memory, global_capacity * sizeof(int));
    init_counter<<<gridSize, blockSize>>>();
    cudaDeviceSynchronize();
    while( host_count_f_as_vpath>0 or host_count_f_ds_vpath>0 or host_count_f_ds_wall>0 or host_count_f_2_saddle>0 or host_count_f_max>0 or host_count_f_min>0 or host_count_f_saddle>0)
    {
        total_ite++;
        cout<<"path:"<<host_count_f_as_vpath<<","<<host_count_f_ds_vpath<<","<<host_count_f_as_wall<<","<<host_count_f_ds_wall<<","<<host_count_f_max<<", "<<host_count_f_min<<", "<<host_count_f_saddle<<endl;
        int cp_ite = 0;
        while(host_count_f_max>0 or host_count_f_2_saddle >0 or host_count_f_min>0 or host_count_f_saddle>0)
        {
            cout<<"wrong:"<<host_count_f_max<<","<< host_count_f_2_saddle <<","<<host_count_f_min<<", "<<host_count_f_saddle<<endl;
            std::vector<float> temp_time;
            cp_ite++;
            float gradient_pairing_sub = 0.0;
            float get_f_cp_sub = 0.0;
            float get_f_path_sub = 0.0;
            float fix_f_cp_sub = 0.0;
            float fix_f_vpath_sub = 0.0;
            float init_cells_sub = 0.0;

            cudaEventRecord(start, 0);
            initializeKernel1<<<gridSize, blockSize>>>();
            cudaDeviceSynchronize();
            

            if(host_count_f_min>0)
            {
                dim3 gridSize_min((host_count_f_min + blockSize.x - 1) / blockSize.x);
                fix_minimum<<<gridSize_min, blockSize>>>(width, height, depth);
                cudaDeviceSynchronize();
            }
            
            if(host_count_f_saddle>0 && preserve_saddles == 1)
            {
                dim3 gridSize_saddle((host_count_f_saddle + blockSize.x - 1) / blockSize.x);
                fix_saddle<<<gridSize_saddle, blockSize>>>(width, height, depth);
                cudaDeviceSynchronize();
            }

            if(host_count_f_2_saddle>0 && preserve_saddles == 1)
            {
                dim3 gridSize_2_saddle((host_count_f_2_saddle + blockSize.x - 1) / blockSize.x);
                fix_2saddle<<<gridSize_2_saddle, blockSize>>>(width, height, depth);
                
                cudaDeviceSynchronize();
            }
            
            if(host_count_f_max>0)
            {
                dim3 gridSize_max((host_count_f_max + blockSize.x - 1) / blockSize.x);
                fix_maximum<<<gridSize_max, blockSize>>>(width, height, depth);
                cudaDeviceSynchronize();
            }

            

            applyDeltaBuffer1<<<gridSize, blockSize>>>(width, height, depth);
            cudaDeviceSynchronize();

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            fix_f_cp+=elapsedTime;
            fix_f_cp_sub+=elapsedTime;
    
            cudaEventRecord(start, 0);
            init_vertices<<<gridSize, blockSize>>>();
            cudaDeviceSynchronize();
            init_edges<<<gridSizeEdge, blockSize>>>();
            cudaDeviceSynchronize();
            init_faces<<<gridSizeFace, blockSize>>>();
            cudaDeviceSynchronize();
            init_tetras<<<gridSizeTetrahedra, blockSize>>>();
            cudaDeviceSynchronize();

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            init_cells+=elapsedTime;
            init_cells_sub+=elapsedTime;
            

            cudaEventRecord(start, 0);
            processVertices<<<gridSize, blockSize>>>(numVertices, width, height, depth, 1);
            cudaDeviceSynchronize();
            edgePairingKernel<<<gridSizeEdge, blockSize>>>(width, height, depth, numVertices, numEdges, numFaces, 1, 0);
            cudaDeviceSynchronize();
            trianglePairingKernel<<<gridSizeFace, blockSize>>>(width, height, depth, numVertices, numEdges, numFaces, numTetrahedras,1, 0);
            cudaDeviceSynchronize();
            
            edgePairingKernel<<<gridSizeEdge, blockSize>>>(width, height, depth, numVertices, numEdges, numFaces, 1 ,1);
            cudaDeviceSynchronize();
            trianglePairingKernel<<<gridSizeFace, blockSize>>>(width, height, depth, numVertices, numEdges, numFaces, numTetrahedras,1, 1);
            cudaDeviceSynchronize();
            

            // if(filtration == 1)
            // {
            //     cout<<"simplification"<<endl;
            //     reverse_as_vpath<<<gridSizeFace, blockSize>>>(width, height, depth,1);
            //     cudaDeviceSynchronize();
                
            //     reverse_ds_vpath<<<gridSizeEdge, blockSize>>>(width, height, depth,1);
            //     cudaDeviceSynchronize();
            // }


            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            gradient_pairing+=elapsedTime;
            gradient_pairing_sub+=elapsedTime;
            
            cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(int));
            cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(int));
            cudaMemcpyToSymbol(count_f_saddle, &initialValue, sizeof(int));
            cudaMemcpyToSymbol(count_f_2_saddle, &initialValue, sizeof(int));

            cudaDeviceSynchronize();

            // get_false_cp
            cudaEventRecord(start, 0);
            get_false_minimum<<<gridSize, blockSize>>>();
            if(preserve_saddles==1)
            {
                get_false_saddle<<<gridSizeEdge, blockSize>>>();
                get_false_2_saddle<<<gridSizeFace, blockSize>>>();
            }
            get_false_maximum<<<gridSizeTetrahedra, blockSize>>>();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            get_f_cp+=elapsedTime;
            get_f_cp_sub+=elapsedTime;

            // finished extract false cp
            cudaDeviceSynchronize();
            cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&host_count_f_saddle, count_f_saddle, sizeof(int), 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&host_count_f_2_saddle, count_f_2_saddle, sizeof(int), 0, cudaMemcpyDeviceToHost);
            
            cudaDeviceSynchronize();
            

            temp_time.push_back(gradient_pairing_sub);

            temp_time.push_back(get_f_cp_sub);
            
            temp_time.push_back(get_f_path_sub);
            
            temp_time.push_back(fix_f_cp_sub);
            
            temp_time.push_back(fix_f_vpath_sub);
            
            temp_time.push_back(init_cells_sub);
            
            temp_time.push_back(cp_ite);
        
            time_counter.push_back(temp_time);
           
        }
        if(preserve_vpath == 0) break;

    
        cudaDeviceSynchronize();
            
        
        checkCudaError(cudaMemcpyToSymbol(count_f_ds_vpath, &initialValue, sizeof(int)),"error 18");
        checkCudaError(cudaMemcpyToSymbol(count_f_as_vpath, &initialValue, sizeof(int)),"error 18");
        checkCudaError(cudaMemcpyToSymbol(count_f_ds_wall, &initialValue, sizeof(int)),"error 18");
        checkCudaError(cudaMemcpyToSymbol(count_f_as_wall, &initialValue, sizeof(int)),"error 18");
        // get false vpath cases;
        // preserve the geometry
        if(preserve_geometry == 1)
        {
            cudaEventRecord(start, 0);
            get_false_as_vpath<<<gridSizeFace, blockSize>>>(width, height, depth);
            cudaDeviceSynchronize();
            get_false_ds_vpath<<<gridSizeEdge, blockSize>>>(width, height, depth);
            cudaDeviceSynchronize();
        }
        else if(preserve_vpath == 1 && preserve_geometry == 0)
        {
            cudaEventRecord(start, 0);
            get_false_as_vpath_connectivity<<<gridSizeFace, blockSize>>>(width, height, depth);
            cudaDeviceSynchronize();
            get_false_ds_vpath_connectivity<<<gridSizeEdge, blockSize>>>(width, height, depth);
            cudaDeviceSynchronize();
        }
        // get_false_as_wall<<<gridSizeEdge, blockSize>>>(width, height, depth);
        init_masks<<<gridSizeFace, blockSize>>>();
        cudaDeviceSynchronize();

        if(preserve_connectors==1)
        {
            get_false_ds_wall<<<gridSize, blockSize, shared_memory_size>>>(width, height, depth, global_memory, global_capacity);
        }   
        

        cudaDeviceSynchronize();
        cudaMemset(global_memory, 0, global_capacity * sizeof(int));
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        get_f_path+=elapsedTime;

        // get false vapth to host;
        checkCudaError(cudaMemcpyFromSymbol(&host_count_f_as_vpath, count_f_as_vpath, sizeof(int), 0, cudaMemcpyDeviceToHost),"error 19");
        checkCudaError(cudaMemcpyFromSymbol(&host_count_f_ds_vpath, count_f_ds_vpath, sizeof(int), 0, cudaMemcpyDeviceToHost),"error 19");
        checkCudaError(cudaMemcpyFromSymbol(&host_count_f_as_wall, count_f_as_wall, sizeof(int), 0, cudaMemcpyDeviceToHost),"error 19");
        checkCudaError(cudaMemcpyFromSymbol(&host_count_f_ds_wall, count_f_ds_wall, sizeof(int), 0, cudaMemcpyDeviceToHost),"error 19");
        cudaDeviceSynchronize();
        cout<<host_count_f_as_vpath<<"," <<host_count_f_ds_vpath<<", "<<host_count_f_ds_wall<<endl;
        // return 0;
        cudaEventRecord(start, 0);
        initializeKernel1<<<gridSize, blockSize>>>();
        cudaDeviceSynchronize();
        if(host_count_f_as_vpath>0){
            dim3 gridSize_as_vpath((host_count_f_as_vpath + blockSize.x - 1) / blockSize.x);
            fix_as_vpath<<<gridSize_as_vpath, blockSize>>>(width, height, depth);
            
            cudaDeviceSynchronize();
        }

        if(host_count_f_ds_vpath>0){
            dim3 gridSize_ds_vpath((host_count_f_ds_vpath + blockSize.x - 1) / blockSize.x);
            fix_ds_vpath<<<gridSize_ds_vpath, blockSize>>>(width, height, depth);
            cudaDeviceSynchronize();
            checkCudaError(cudaMemcpyFromSymbol(&host_count_f_ds_vpath, count_f_ds_vpath, sizeof(int), 0, cudaMemcpyDeviceToHost), "errorhere1");
        }

        if(host_count_f_ds_wall>0 && preserve_connectors==1){
            dim3 gridSize_ds_wall((host_count_f_ds_wall + blockSize.x - 1) / blockSize.x);
            fix_ds_wall<<<gridSize_ds_wall, blockSize>>>(width, height, depth);
            cudaDeviceSynchronize();
        }
        
        applyDeltaBuffer1<<<gridSize, blockSize>>>(width, height, depth);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        fix_f_vpath+=elapsedTime;


        cudaEventRecord(start, 0);
        init_vertices<<<gridSize, blockSize>>>();
        init_edges<<<gridSizeEdge, blockSize>>>();
        init_faces<<<gridSizeFace, blockSize>>>();
        init_tetras<<<gridSizeTetrahedra, blockSize>>>();
    
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        init_cells+=elapsedTime;
        // reassign gradient for the current data;

        cudaEventRecord(start, 0);
        processVertices<<<gridSize, blockSize>>>(numVertices, width, height, depth, 1);
        cudaDeviceSynchronize();
        edgePairingKernel<<<gridSizeEdge, blockSize>>>(width, height, depth, numVertices, numEdges, numFaces, 1, 0);
        cudaDeviceSynchronize();
        trianglePairingKernel<<<gridSizeFace, blockSize>>>(width, height, depth, numVertices, numEdges, numFaces, numTetrahedras,1, 0);
        cudaDeviceSynchronize();
        
        
       
        edgePairingKernel<<<gridSizeEdge, blockSize>>>(width, height, depth, numVertices, numEdges, numFaces, 1 ,1);
        cudaDeviceSynchronize();
        trianglePairingKernel<<<gridSizeFace, blockSize>>>(width, height, depth, numVertices, numEdges, numFaces, numTetrahedras,1, 1);
        cudaDeviceSynchronize();
        

        // if(filtration == 1)
        // {
        //     reverse_as_vpath<<<gridSizeFace, blockSize>>>(width, height, depth,1);
        //     cudaDeviceSynchronize();
        //     reverse_ds_vpath<<<gridSizeEdge, blockSize>>>(width, height, depth,1);
        //     cudaDeviceSynchronize();
        // }

        // initialize count_f_variables;

        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        gradient_pairing+=elapsedTime;

        cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(int));
        cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(int));
        cudaMemcpyToSymbol(count_f_saddle, &initialValue, sizeof(int));
        cudaMemcpyToSymbol(count_f_2_saddle, &initialValue, sizeof(int));
        cudaMemcpyToSymbol(count_f_ds_vpath, &initialValue, sizeof(int));
        cudaMemcpyToSymbol(count_f_as_vpath, &initialValue, sizeof(int));
        cudaMemcpyToSymbol(count_f_ds_wall, &initialValue, sizeof(int));
        cudaMemcpyToSymbol(count_f_as_wall, &initialValue, sizeof(int));
        
        // get false critical points;
        cudaEventRecord(start, 0);
        cudaDeviceSynchronize();
        get_false_minimum<<<gridSize, blockSize>>>();
        get_false_saddle<<<gridSizeEdge, blockSize>>>();
        get_false_2_saddle<<<gridSizeFace, blockSize>>>();
        get_false_maximum<<<gridSizeTetrahedra, blockSize>>>();
        
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        get_f_cp+=elapsedTime;
        
        // translate number of f_cp to host;
        cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&host_count_f_saddle, count_f_saddle, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&host_count_f_2_saddle, count_f_2_saddle, sizeof(int), 0, cudaMemcpyDeviceToHost);
        

        // if no f_cp, then get false vpath;
        if(host_count_f_max==0 and host_count_f_min==0 and host_count_f_saddle ==0 and host_count_f_2_saddle ==0)
        {
            cudaEventRecord(start, 0);
            if(preserve_geometry == 1)
            {
                cudaEventRecord(start, 0);
                get_false_as_vpath<<<gridSizeFace, blockSize>>>(width, height, depth);
                cudaDeviceSynchronize();
                get_false_ds_vpath<<<gridSizeEdge, blockSize>>>(width, height, depth);
                cudaDeviceSynchronize();
            }
            else if(preserve_vpath == 1 && preserve_geometry == 0)
            {
                cudaEventRecord(start, 0);
                get_false_as_vpath_connectivity<<<gridSizeFace, blockSize>>>(width, height, depth);
                cudaDeviceSynchronize();
                get_false_ds_vpath_connectivity<<<gridSizeEdge, blockSize>>>(width, height, depth);
                cudaDeviceSynchronize();
            }
            
            // get_false_as_wall<<<gridSizeEdge, blockSize>>>(width, height, depth);
            init_masks<<<gridSizeFace, blockSize>>>();
            cudaDeviceSynchronize();
            if(preserve_connectors == 1)
            {
                get_false_ds_wall<<<gridSize, blockSize, shared_memory_size>>>(width, height, depth, global_memory, global_capacity);
            }
            
            cudaDeviceSynchronize();
            cudaMemset(global_memory, 0, global_capacity * sizeof(int));
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            get_f_path+=elapsedTime;
        }
        
        // translate number of false vpath to host;
        checkCudaError(cudaMemcpyFromSymbol(&host_count_f_ds_vpath, count_f_ds_vpath, sizeof(int), 0, cudaMemcpyDeviceToHost), "errorhere");
        checkCudaError(cudaMemcpyFromSymbol(&host_count_f_as_vpath, count_f_as_vpath, sizeof(int), 0, cudaMemcpyDeviceToHost), "errorhere");
        checkCudaError(cudaMemcpyFromSymbol(&host_count_f_ds_wall, count_f_ds_wall, sizeof(int), 0, cudaMemcpyDeviceToHost), "errorhere");
        checkCudaError(cudaMemcpyFromSymbol(&host_count_f_as_wall, count_f_as_wall, sizeof(int), 0, cudaMemcpyDeviceToHost), "errorhere");

        cudaDeviceSynchronize();
        cout<<"wrong wall:"<<host_count_f_as_wall << ", "<<host_count_f_ds_wall<<endl;
        cout<<host_count_f_max<<", "<< host_count_f_2_saddle <<","<<host_count_f_min<<", "<<host_count_f_saddle<<endl;
        
    }
   
    cudaFree(global_memory);
    end = std::chrono::high_resolution_clock::now();
    duration = end - startt;

    additional_time = duration.count();
    checkCudaError(cudaMemcpy(decp_data1.data(), decp_temp, numVertices * sizeof(double), cudaMemcpyDeviceToHost), "error 12");
    cudaDeviceSynchronize();
    
    // cout<<"wrong wall:"<<host_count_f_as_wall<<", "<<host_count_f_ds_wall<<endl;
    
    int c = 0;
    for(int i=0;i<size2;i++)
    {
        if(decp_data_copy[i]!=decp_data1[i]) c++;
    }
    saveVectorToBin(decp_data1, fix_path);

    std::vector<int> delta_counter1(size2);
    int* counter_device;
    cudaMemcpyFromSymbol(&counter_device, delta_counter, sizeof(int*), 0, cudaMemcpyDeviceToHost);

    
    cudaMemcpy(delta_counter1.data(), counter_device, size2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    if(edit_type1==1)
    {
        cost(filename, decp_data1, decp_data_copy, input_data1, compressor_id, delta_counter1);
    }
    else
    {
        original_cost(filename, decp_data1, decp_data_copy, input_data1, compressor_id);
    }
    
    cudaDeviceSynchronize();
    cout<<c<<endl;
    cout<<"build_cells: "<<build_cells/1000<<" ,gradient_pairing: "<<gradient_pairing/1000<<",get_f_cp: "<<get_f_cp/1000
    <<" ,get_f_path: "<<get_f_path/1000 <<" ,fix_f_cp: "<<fix_f_cp/1000<<" ,fix_f_vpath: "<<fix_f_vpath/1000<<", init_cells: "<<init_cells/1000<<endl;
    

    std::ofstream outFilep("./stat_result/performance1_cuda_"+filename+"_"+std::to_string(bound1)+"_"+compressor_id+".txt", std::ios::app);
        // 检查文件是否成功打开
        if (!outFilep) {
            std::cerr << "Unable to open file for writing." << std::endl;
            return 1; // 返回错误码
        }
        // finddirection:0, getfcp:1,  mappath2, fixcp:3
        
        // outFilep << std::to_string(number_of_thread)<<":" << std::endl;
        // outFilep << "duration: "<<duration1.count() << std::endl;
        outFilep << std::setprecision(17)<< "related_error: "<<range1 << std::endl;
        outFilep << "preserve_saddles: "<< preserve_saddles << std::endl;
        outFilep << "preserve_vpath: "<< preserve_vpath << std::endl;
        outFilep << "preserve_geometry: "<< preserve_geometry << std::endl;
        outFilep << "filtration: "<< filtration << std::endl;
        outFilep << "preserve_connectors: "<< preserve_connectors << std::endl;
        outFilep << "preserve_types: "<< preserve_type << std::endl;
        outFilep << "build_cells: "<<build_cells/1000 << std::endl;
        
        outFilep << "gradient_pairing: "<<gradient_pairing/1000 << std::endl;
        
        outFilep << "get_f_cp: " << get_f_cp/1000 << std::endl;
        outFilep << "get_f_path: " << get_f_path/1000 << std::endl;
        
        outFilep << "fix_f_cp: " << fix_f_cp/1000 << std::endl;
        
        outFilep << "fix_f_vpath:" << fix_f_vpath/ 1000<< std::endl;
        outFilep << "init_cells:" << init_cells/1000<< std::endl;
        
        outFilep << "iteration number:" << total_ite+1 << std::endl;
        // outFilep << "edit_ratio: "<< ratio << std::endl;  
        int c1 = 0;  
        for (const auto& row : time_counter) {
            outFilep << "iteration: "<<c1<<": ";
            for (size_t i = 0; i < row.size(); ++i) {
                outFilep << row[i];
                if (i != row.size() - 1) { // 不在行的末尾时添加逗号
                    outFilep << ", ";
                }
            }
            // 每写完一行后换行
            outFilep << std::endl;
            c1+=1;
        }
        outFilep << "\n"<< std::endl;

    // // get_cp_number<<<1,1>>>();
    // cudaDeviceSynchronize();
    
    // float get_f_path = 0.0;
    // float fix_f_cp = 0.0;
    // float fix_f_vpath = 0.0;
    // // float init_cells = 0.0;
    // return 0;
    return 0;
    
    
}

