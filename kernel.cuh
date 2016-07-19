#ifndef KERNEL_H
#define KERNEL_H
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define hostList double* h_vertices, int h_n_vertices, bool* h_vertex_removed, double* h_quadrics, int* h_faces, int* h_vert_face_header, int* h_vert_face_data, bool* h_face_removed, int* h_edges, double* h_edge_cost, int* h_edge_queue, int* h_vert_edge_from_header, int* h_vert_edge_from_data, int* h_vert_edge_to_header, int* h_vert_edge_to_data, bool* h_edge_removed, int h_n_cells, int* h_vertex_in_cell, int* h_initial_vertices, int* h_cell_vertices, int* h_cell_vertices_size, int* h_cell_heap, int* h_cell_heap_size, int h_n_edges, int h_n_faces
#define hostArgumentList h_vertices , n_vertices, h_vertex_removed , h_quadrics , h_faces , h_vert_face_header , h_vert_face_data , h_face_removed , h_edges , h_edge_cost , h_edge_queue , h_vert_edge_from_header , h_vert_edge_from_data , h_vert_edge_to_header , h_vert_edge_to_data , h_edge_removed , n_cells, h_vertex_in_cell , h_initial_vertices , h_cell_vertices , h_cell_vertices_size , h_cell_heap , h_cell_heap_size , n_edges, n_faces


//Device global variables
#define environmentList double* d_vertices, int d_n_vertices, bool* d_vertex_removed, double* d_quadrics, int* d_faces, int* d_vert_face_header, int* d_vert_face_data, bool* d_face_removed, int* d_edges, double* d_edge_cost, int* d_edge_queue, int* d_vert_edge_from_header, int* d_vert_edge_from_data, int* d_vert_edge_to_header, int* d_vert_edge_to_data, bool* d_edge_removed, int d_n_cells, int* d_vertex_in_cell, int* d_initial_vertices, int* d_cell_vertices, int* d_cell_vertices_size, int* d_cell_heap, int* d_cell_heap_size, int d_n_edges, int d_n_faces
#define environmentReferenceList double*& d_vertices, int& d_n_vertices, bool*& d_vertex_removed, double*& d_quadrics, int*& d_faces, int*& d_vert_face_header, int*& d_vert_face_data, bool*& d_face_removed, int*& d_edges, double*& d_edge_cost, int*& d_edge_queue, int*& d_vert_edge_from_header, int*& d_vert_edge_from_data, int*& d_vert_edge_to_header, int*& d_vert_edge_to_data, bool*& d_edge_removed, int& d_n_cells, int*& d_vertex_in_cell, int*& d_initial_vertices, int*& d_cell_vertices, int*& d_cell_vertices_size, int*& d_cell_heap, int*& d_cell_heap_size, int& d_n_edges, int& d_n_faces
#define environmentArgumentList d_vertices, d_n_vertices, d_vertex_removed, d_quadrics, d_faces, d_vert_face_header, d_vert_face_data, d_face_removed, d_edges, d_edge_cost, d_edge_queue, d_vert_edge_from_header, d_vert_edge_from_data, d_vert_edge_to_header, d_vert_edge_to_data, d_edge_removed, d_n_cells, d_vertex_in_cell, d_initial_vertices, d_cell_vertices, d_cell_vertices_size, d_cell_heap, d_cell_heap_size, d_n_edges, d_n_faces

void initDeviceEnvironment(hostList, environmentReferenceList);


#endif
