#include "kernel.cuh"
#include <iostream>

//get position from header array
//Size of each structure on arrays
#define HEADER_SIZE  3
#define FACE_SIZE 3
#define VERTEX_SIZE 3
#define FACE_DATA_BATCH_SIZE 50
#define EDGE_DATA_BATCH_SIZE 50
#define QUADRIC_SIZE 16 //Quadric for a vertex is a 4x4 matrix
#define EDGE_SIZE 2

//UGRID
#define CELL_HEADER_SIZE 2

//ACCESS Vertex
#define getFaceVertexId(face,vertex) faces[FACE_SIZE*face+vertex]
#define getFaceHeaderPos(vertex) vert_face_header[HEADER_SIZE*vertex]
#define getFaceCurrSize(vertex) vert_face_header[HEADER_SIZE*vertex+1]
#define faceIncreaseSize(vertex) vert_face_header[HEADER_SIZE*vertex+1]++
#define faceDecreaseSize(vertex) vert_face_header[HEADER_SIZE*vertex+1]--
#define getFaceId(vid,p) vert_face_data[getFaceHeaderPos(vid)+p]
#define getX(vid) vertices[VERTEX_SIZE*vid]
#define getY(vid) vertices[VERTEX_SIZE*vid+1]
#define getZ(vid) vertices[VERTEX_SIZE*vid+2]

//ACCESS TO EDGES
#define getEdgeVertexId(edge,vid) edges[EDGE_SIZE*edge+vid] //Get vertex id (0 or 1) of edge
#define getEdgeFromHeaderPos(vid) vert_edge_from_header[HEADER_SIZE*vid]
#define getEdgeFromCurrSize(vid) vert_edge_from_header[HEADER_SIZE*vid+1]
#define edgeFromIncreaseSize(vid) assert(vert_edge_from_header[HEADER_SIZE*vid+1] < EDGE_DATA_BATCH_SIZE); vert_edge_from_header[HEADER_SIZE*vid+1]++
#define edgeFromDecreaseSize(vid) vert_edge_from_header[HEADER_SIZE*vid+1]--
#define getEdgeFromDataId(vid,p) vert_edge_from_data[getEdgeFromHeaderPos(vid)+p]

//edge_to
#define edgeToIncreaseSize(vid) assert(vert_edge_from_header[HEADER_SIZE*vid+1] < EDGE_DATA_BATCH_SIZE); vert_edge_to_header[HEADER_SIZE*vid+1]++
#define edgeToDecreaseSize(vid) vert_edge_to_header[HEADER_SIZE*vid+1]--
#define getEdgeToHeaderPos(vid) vert_edge_to_header[HEADER_SIZE*vid]
#define getEdgeToCurrSize(vid) vert_edge_to_header[HEADER_SIZE*vid+1]
#define getEdgeToDataId(vid,p) vert_edge_to_data[getEdgeToHeaderPos(vid)+p]

//Access
#define getPlacementX(vid1,vid2) (getX(vid1)+getX(vid2))/2
#define getPlacementY(vid1,vid2) (getY(vid1)+getY(vid2))/2
#define getPlacementZ(vid1,vid2) (getZ(vid1)+getZ(vid2))/2

//UNIFORM GRID
#define getCellHeaderPos(cell) cell_header[CELL_HEADER_SIZE*cell]
#define getCellHeaderSize(cell) cell_header[CELL_HEADER_SIZE*cell+1]
#define increaseCellSize(cell) cell_header[CELL_HEADER_SIZE*cell+1]++
#define getCellVertexId(cell,p) cell_vertices[cell*n_vertices+p]
#define HEAP_HEADER_SIZE 3
#define HEAP_SIZE_VERTEX_RATIO 8
#define getHeapHead(cell) cell_heap_header[HEAP_HEADER_SIZE*(cell)]
#define getCellHeapSize(cell) cell_heap_header[HEAP_HEADER_SIZE*(cell)+1]
#define getCellMaxHeapSize(cell) cell_heap_header[HEAP_HEADER_SIZE*(cell)+2]


//======DEVICE (GLOBAL) VARIABLES======
//VERTICES
__device__ double* vertices;
__device__ bool* vertex_removed;
__device__ double* quadrics;

//FACES
__device__ int* faces;
__device__ int* vert_face_header;
__device__ int* vert_face_data;
__device__ bool* face_removed;

//EDGES
__device__ int* edges;
__device__ double* edge_cost;
__device__ int* edge_queue;
__device__ int* vert_edge_from_header;
__device__ int* vert_edge_from_data;
__device__ int* vert_edge_to_header;
__device__ int* vert_edge_to_data;
__device__ bool* edge_removed;

//UNIFORM GRID
__device__ int n_cells;

//VERTEX
__device__ int* cell_data;
__device__ int* cell_header;
__device__ int* vertex_in_cell;
__device__ int* initial_vertices;
__device__ int* cell_vertices;
__device__ int* cell_vertices_size;

//EDGE
__device__ int* cell_heap;
__device__ int* cell_heap_header;
__device__ elem* cell_heap_data;
__device__ int* cell_heap_size;


//UNIFORM GRID DIM AND OFFSET
__device__ double dim[3];
__device__ double offset[3];
__device__ int grid_res;
__device__ double* bbox;


//NO.
__device__ int n_edges;
__device__ int n_faces;
__device__ int n_vertices;

//=====



//=====HEAP FUNCTIONS ======

__device__ bool d_compare(const elem &left,const elem &right)
{
  return left.cost < right.cost;
}

//=====HEAP FUNCTIONS ======

//Move node up
__device__ void d_percUP(int i, elem* heap)
{
  while(i/2 > 0)
  {
    if(d_compare(heap[i], heap[i/2]))
    {
      elem aux = heap[i];
      heap[i] = heap[i/2];
      heap[i/2] = aux;
    }
    i/=2;
  }
}

//d_insert element e into heap of size n
__device__ void d_insert(elem e, elem* heap, int& n)
{
  // cout << "Elem id: " << e.id << endl;
  heap[n+1] = e;
  n++;
  d_percUP(n, heap);

}

//Find child of least cost
__device__ int d_minChild(int i, elem* heap, int n)
{
  //Does not have right child
  if (i * 2 + 1 > n) return i*2;
  else{
    if(d_compare(heap[2*i],heap[2*i+1])) return 2*i;
    else return 2*i+1;
  }
}

//Move node down the heap
__device__ void d_percDown(int i, elem* heap, int n)
{
  while(i*2 <= n)
  {
    int mc = d_minChild(i, heap, n);
    if(d_compare(heap[mc],heap[i]))
    {
      elem tmp = heap[mc];
      heap[mc] = heap[i];
      heap[i] = tmp;
    }
     i = mc;
  }
}

//Pop element of least value
__device__ elem d_pop(elem* heap, int& n)
{
  elem ret = heap[1];
  heap[1] = heap[n];
  n--;
  d_percDown(1, heap, n);
  return ret;
}

//Turn vector into heap or update heap
__device__ void d_heapify(elem* vec, int n)
{
  int i = n/2;
  while( i > 0)
  {
    d_percDown(i,vec,n);
    i--;
  }
}


//===END OF HEAP===



__device__ bool d_isEntirelyInCell(int eid)
{
  //Return if both endpoints are withing the same cell
  return vertex_in_cell[getEdgeVertexId(eid,0)] == vertex_in_cell[getEdgeVertexId(eid,1)];
}

__device__ bool d_isCrownInCell(int vid)
{
  //Test edges leaving (from) vid to see if their endpoints also lie within vid's cell
  //printf("Vertex %d\n", vid);
  for(int i = 0 ; i < getEdgeFromCurrSize(vid); ++i)
  {
    //printf("From\n");
    //Check edge i from vid

  //  printf("edges from %d\n", getEdgeFromCurrSize(vid));
    int eid = vert_edge_from_data[EDGE_DATA_BATCH_SIZE*vid+i];
    //int eid = getEdgeFromDataId(vid,i);
    int endp = getEdgeVertexId(eid,1);

    if(vertex_in_cell[endp] != vertex_in_cell[vid])
      return false;
  }

  //Test edges arriving (to) vid
  for(int i = 0 ; i < getEdgeToCurrSize(vid); ++i)
  {
    //printf("To\n");
    //int eid = getEdgeToDataId(vid,i);
    int eid = vert_edge_to_data[EDGE_DATA_BATCH_SIZE*vid+i];
    int endp = getEdgeVertexId(eid,0);
    if(vertex_in_cell[endp] != vertex_in_cell[vid])
      return false;
  }
  return true;
}


__global__ void computeGridEdges()
{

  int i = threadIdx.x;

  printf("Thread %d\n", i);

  for(int j = 0; j < cell_vertices_size[i]; ++j)
  {
    //printf("j | cell_vertices_size %d | %d\n", j, cell_vertices_size[i]);
    for(int k = 0; k < getEdgeFromCurrSize(getCellVertexId(i,j)); ++k)
    {

      int vid = getCellVertexId(i,j);
      //int eid = getEdgeFromDataId(getCellVertexId(i,j),k);
      int eid = vert_edge_from_data[EDGE_DATA_BATCH_SIZE*vid+k];

      if(d_isEntirelyInCell(eid) && d_isCrownInCell(getEdgeVertexId(eid,0)) && d_isCrownInCell(getEdgeVertexId(eid,1)))
      {

        elem temp = {eid,edge_cost[eid]};
        //d_insert(temp, cell_heap_data+getHeapHead(i), getCellHeapSize(i));
      }

    }
  }



}

__global__ void initUniformGrid()
{

  printf("grid_res %d\n", grid_res);
  n_cells = grid_res*grid_res;
  n_cells = n_cells*grid_res;
  offset[0] = bbox[0];
  offset[1] = bbox[1];
  offset[2] = bbox[2];

  dim[0] = bbox[3]/grid_res;
  dim[1] = bbox[4]/grid_res;
  dim[2] = bbox[5]/grid_res;

  for(int i = 0; i < n_cells; ++i)
  {
    initial_vertices[i] = 0;
    cell_vertices_size[i] = 0;

    getHeapHead(i) = 0;
    getCellHeapSize(i) = 0;
  }

  for(int i = 0; i < n_vertices; ++i)
  {
    if(vertex_removed[i])
    {
      continue;
    }
    int cx = (getX(i) - offset[0])/dim[0];
    cx -= cx/grid_res;
    int cy = (getY(i) - offset[1])/dim[1];
    cy -= cy/grid_res;
    long long cz = (getZ(i) - offset[2])/dim[2];
    cz -= cz/grid_res;
    int cpos = cx + grid_res*cy + grid_res*grid_res*cz;

    vertex_in_cell[i] = cpos;
    initial_vertices[cpos]++;

    cell_vertices[cpos*n_vertices+cell_vertices_size[cpos]] = i;
    cell_vertices_size[cpos]++;
  }

  for(int i = 0; i < n_cells; ++i)
  {
    getCellHeapSize(i) = 0;
    getCellMaxHeapSize(i) = HEAP_SIZE_VERTEX_RATIO*initial_vertices[i];

    if(i!=0){
      getHeapHead(i) = getHeapHead(i-1) + getCellMaxHeapSize(i-1)+1;

    }
    else{
      getHeapHead(i) = 0;
    }
  }

  //Calculate edges for each cell separately
  computeGridEdges<<<1,n_cells>>>();

}


//Set pointers of global variables
__global__ void initDevice(environmentList)
{
  vertices = d_vertices;
  vertex_removed = d_vertex_removed;
  quadrics = d_quadrics;
  faces = d_faces;
  vert_face_header = d_vert_face_header;
  vert_face_data = d_vert_face_data;
  face_removed = d_face_removed;
  edges = d_edges;
  edge_cost = d_edge_cost;
  edge_queue = d_edge_queue;
  vert_edge_from_header = d_vert_edge_from_header;
  vert_edge_from_data = d_vert_edge_from_data;
  vert_edge_to_header = d_vert_edge_to_header;
  vert_edge_to_data = d_vert_edge_to_data;
  edge_removed = d_edge_removed;
  vertex_in_cell = d_vertex_in_cell;
  initial_vertices = d_initial_vertices;
  cell_vertices = d_cell_vertices;
  cell_vertices_size = d_cell_vertices_size;
  cell_heap_header = d_cell_heap_header;
  cell_heap_data = d_cell_heap_data;
  cell_heap_size = d_cell_heap_size;
  bbox = d_bbox;
  n_vertices = d_n_sizes[0];
  n_faces = d_n_sizes[1];
  n_edges = d_n_sizes[2];
  grid_res = d_n_sizes[3];
}

void initDeviceEnvironment(hostList,environmentReferenceList)
{

  //INTEGERS
  d_n_edges = h_n_edges;
  d_n_faces = h_n_faces;
  d_n_cells = h_n_cells;
  d_n_vertices = h_n_vertices;

  //INIT_DATA_STRUCTURES
  int size = FACE_SIZE*h_n_faces*sizeof(int);
  cudaMalloc(&d_faces,size);
  cudaMemcpy(d_faces, h_faces, size, cudaMemcpyHostToDevice);

  size = h_n_faces*sizeof(bool);
  cudaMalloc(&d_face_removed,size);
  cudaMemcpy(d_face_removed, h_face_removed, size, cudaMemcpyHostToDevice);

  size = VERTEX_SIZE*h_n_vertices*sizeof(double);
  cudaMalloc(&d_vertices, size);
  cudaMemcpy(d_vertices, h_vertices, size, cudaMemcpyHostToDevice);

  size = h_n_vertices*sizeof(bool);
  cudaMalloc(&d_vertex_removed, size);
  cudaMemcpy(d_vertex_removed, h_vertex_removed, size, cudaMemcpyHostToDevice);

  size = 16*h_n_vertices*sizeof(double);
  cudaMalloc(&d_quadrics,size);
  cudaMemcpy(d_quadrics, h_quadrics, size, cudaMemcpyHostToDevice);

  size = HEADER_SIZE*h_n_vertices*sizeof(int);
  cudaMalloc(&d_vert_face_header, size);
  cudaMemcpy(d_vert_face_header, h_vert_face_header, size, cudaMemcpyHostToDevice);

  size = FACE_DATA_BATCH_SIZE*h_n_vertices*sizeof(int);
  cudaMalloc(&d_vert_face_data, size);
  cudaMemcpy(d_vert_face_data, h_vert_face_data, size, cudaMemcpyHostToDevice);

  //INIT_EDGES
  size = h_n_faces*6*sizeof(int);
  cudaMalloc(&d_edges, size);
  cudaMemcpy(d_edges, h_edges, size, cudaMemcpyHostToDevice);

  size = HEADER_SIZE*h_n_vertices*sizeof(int);
  cudaMalloc(&d_vert_edge_from_header, size);
  cudaMemcpy(d_vert_edge_from_header, h_vert_edge_from_header, size, cudaMemcpyHostToDevice);

  size = EDGE_DATA_BATCH_SIZE*h_n_vertices*sizeof(int);
  cudaMalloc(&d_vert_edge_from_data, size);
  cudaMemcpy(d_vert_edge_from_data, h_vert_edge_from_data, size, cudaMemcpyHostToDevice);

  size = HEADER_SIZE*h_n_vertices*sizeof(int);
  cudaMalloc(&d_vert_edge_to_header, size);
  cudaMemcpy(d_vert_edge_to_header, h_vert_edge_to_header, size, cudaMemcpyHostToDevice);

  size = EDGE_DATA_BATCH_SIZE*h_n_vertices*sizeof(int);
  cudaMalloc(&d_vert_edge_to_data, size);
  cudaMemcpy(d_vert_edge_to_data, h_vert_edge_to_data, size, cudaMemcpyHostToDevice);

  size = h_n_edges*sizeof(double);
  cudaMalloc(&d_edge_cost, size);
  cudaMemcpy(d_edge_cost, h_edge_cost, size, cudaMemcpyHostToDevice);

  size = h_n_edges*sizeof(bool);
  cudaMalloc(&d_edge_removed, size);
  cudaMemcpy(d_edge_removed, h_edge_removed, size, cudaMemcpyHostToDevice);

  //UNIFORM_GRID
  size = h_n_vertices*sizeof(int);
  cudaMalloc(&d_vertex_in_cell, size);
  cudaMemcpy(d_vertex_in_cell, h_vertex_in_cell, size, cudaMemcpyHostToDevice);

  size = h_n_cells*sizeof(int);
  cudaMalloc(&d_initial_vertices, size);
  cudaMemcpy(d_initial_vertices, h_initial_vertices, size, cudaMemcpyHostToDevice);

  size = h_n_cells*sizeof(int);
  cudaMalloc(&d_cell_vertices_size, size);
  cudaMemcpy(d_cell_vertices_size, h_cell_vertices_size, size, cudaMemcpyHostToDevice);

  size = h_n_cells*h_n_vertices*sizeof(int);
  cudaMalloc(&d_cell_vertices, size);
  cudaMemcpy(d_cell_vertices, h_cell_vertices, size, cudaMemcpyHostToDevice);

  size = h_n_cells*(HEADER_SIZE)*sizeof(int);
  cudaMalloc(&d_cell_heap_header, size);
  cudaMemcpy(d_cell_heap_header, h_cell_heap_header, size, cudaMemcpyHostToDevice);

  size = h_n_vertices*sizeof(elem)*HEAP_SIZE_VERTEX_RATIO;
  cudaMalloc(&d_cell_heap_data, size);
  cudaMemcpy(d_cell_heap_data, h_cell_heap_data, size, cudaMemcpyHostToDevice);

  size = h_n_cells*sizeof(int);
  cudaMalloc(&d_cell_heap_size, size);
  cudaMemcpy(d_cell_heap_size, h_cell_heap_size, size, cudaMemcpyHostToDevice);

  //bbox
  size = 6*sizeof(double);
  cudaMalloc(&d_bbox, size);
  cudaMemcpy(d_bbox, h_bbox, size, cudaMemcpyHostToDevice);

  //sizes
  size = 4*sizeof(int);
  cudaMalloc(&d_n_sizes, size);
  cudaMemcpy(d_n_sizes, h_n_sizes, size, cudaMemcpyHostToDevice);

  initDevice<<<1,1>>>(environmentArgumentList);
  cudaDeviceSynchronize();
}

void pullFromDevice(hostList, environmentReferenceList)
{

  std::cerr << "Pulling from device...\n";

  int size = FACE_SIZE*h_n_faces*sizeof(int);
  cudaMemcpy(h_faces, d_faces, size, cudaMemcpyDeviceToHost);

  size = h_n_faces*sizeof(bool);
  cudaMemcpy(h_face_removed, d_face_removed, size, cudaMemcpyDeviceToHost);

  size = VERTEX_SIZE*h_n_vertices*sizeof(double);
  cudaMemcpy(h_vertices, d_vertices, size, cudaMemcpyDeviceToHost);

  size = h_n_vertices*sizeof(bool);
  cudaMemcpy(h_vertex_removed, d_vertex_removed, size, cudaMemcpyDeviceToHost);

  size = HEADER_SIZE*h_n_vertices*sizeof(int);
  cudaMemcpy(h_vert_face_header, d_vert_face_header, size, cudaMemcpyDeviceToHost);
}

void initializeUniformGrid()
{
  std::cerr << "Initializing uniform grid (device)...\n";
  initUniformGrid<<<1,1>>>();
  cudaDeviceSynchronize();
}

void freeDevice(int* d_a)
{
  cudaFree(d_a);
}
