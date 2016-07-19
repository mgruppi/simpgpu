#ifndef _SIMPGPU_H__
#define _SIMPGPU_H__
#include "Surface.h"
#include <iostream>
#include "Classes.h"
//#include "kernel.h"
using namespace std;

class SimpGPU
{

public:
  SimpGPU(Surface* so);

  //init
  void initDataStructures();
  void initEdges();
  void initQuadrics();
  void initUniformGrid();

  //Simplification
  //double getCost(int eid); //Get cost form edge eid
  double updateCosts(int vid);//Update costs for vid's edges
  void simplify(int, int);

  //Etc
  void updateSurface();

  //Host
  Surface* s;
};

#endif // _SIMPGPU_H__
