#include <iostream>
#include <stdlib.h>
#include <string>
#include "Surface.h"
//These methods must be updated to match new data structures Vertex3f and Face3f
#include "SimpVertexClustering.h"
#include "SimpELEN.h"
#include "SimpQEM.h"
#include "SimpGPU.h"

using namespace std;


void writeOutput(char* fpath, Surface* s, string method, float goal)
{
  string sa(fpath);
  string sub = sa.substr(0, sa.length()-4);
  int percentage = goal*100;
  string qtd = to_string(percentage);
  string output = sub+qtd+"_"+method+".off";
  cerr << "Writing output to " + output << endl;
  s->saveOFF(output);
  //s->dumpBoundingBox();
  cerr << deftty;
}

int main(int argc, char** argv)
{
  if(argc < 6)
  {
    cerr << "*USAGE: Simplify <input file> <% of points to remove> <method (elen/qem/gpu/vc)> <grid_resolution> <no of threads>.\n";
    exit(1);
  }

  string method = argv[3];

  if(method != "elen" && method != "qem" && method != "vc" && method != "gpu")
  {
    cerr << "ERROR: Invalid decimation method.\n";
    exit(1);
  }

  Surface* s = new Surface(argv[1]);
  float goal = atof(argv[2]);
  int gridresolution = atoi(argv[4]);
  int nthreads = atoi(argv[5]);

  if(method == "elen")
  {
    method = "ELEN";
    int goal_vertices = goal*s->m_points.size();
    SimpELEN* elen = new SimpELEN(s, nthreads);
    elen->simplify(goal_vertices,gridresolution);
    writeOutput(argv[1],s,method,goal);
  }
  else if (method == "qem")
  {
    method = "QEM";
    int goal_vertices = goal*s->m_points.size();
    SimpQEM* qem = new SimpQEM(s,nthreads);
    qem->simplify(goal_vertices,gridresolution);
    writeOutput(argv[1],s,method,goal);
  }
  else if (method == "gpu")
  {
    method = "GPU";
    int goal_vertices = goal*s->m_points.size();
    SimpGPU* gpusimp = new SimpGPU(s);
    gpusimp->simplify(goal_vertices,gridresolution);
    writeOutput(argv[1],s,method,goal);

  }
  else if (method == "vc")
  {
    method = "VCLUSTERING";
    cout << "Vertex Clustering not available yet.\n";
    exit(1);
    //SimpVertexClustering* vc = new SimpVertexClustering(s, 5);
    //vc->initCells();
    //vc->simplifyClusters();
  }

  //delete vc;
  delete s;

  return 0;
}
