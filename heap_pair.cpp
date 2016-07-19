#include <iostream>
#include <cstdlib>

using namespace std;

struct elem {
  int id;
  float cost;
};

ostream& operator<<(ostream &out, const elem &e) {
  out << "(" << e.id << "," << e.cost << ")";
  return out;
}

bool compare(elem &a, elem &b)
{
  return a.cost < b.cost;
}

void percUp(int i, elem* heap)
{
  while(i/2 > 0)
  {
    if(compare(heap[i], heap[i/2]))
    {
      elem aux = heap[i];
      heap[i] = heap[i/2];
      heap[i/2] = aux;
    }
    i/=2;
  }
}

void insert(elem e, elem* heap, int& n)
{
  heap[n+1] = e;
  n++;
  percUp(n, heap);
}

int minChild(int i, elem* heap, int n)
{
  //Does not have right child
  if (i * 2 + 1 > n) return i*2;
  else{
    if(compare(heap[2*i],heap[2*i+1])) return 2*i;
    else return 2*i+1;
  }
}

void percDown(int i, elem* heap, int n)
{
  while(i*2 <= n)
  {
    int mc = minChild(i, heap, n);
    if(compare(heap[mc],heap[i]))
    {
      elem tmp = heap[mc];
      heap[mc] = heap[i];
      heap[i] = tmp;
    }
     i = mc;
  }
}

elem pop(elem* heap, int& n)
{
  elem ret = heap[1];
  heap[1] = heap[n];
  n--;
  percDown(1, heap, n);
  return ret;
}



void heapify(elem* vec, int n)
{
  int i = n/2;
  while( i > 0)
  {
    percDown(i,vec,n);
    i--;
  }
}

int main()
{
  elem vec[50] = {{400,400},{3,60},{7,54},{9,27},{100,32},{110,13},{52,2},{46,8},{17,25},{2,43},{12,23},{27,27},{35,33},{43,28},{6,10},{73,9},{1,22},{18,21},{57,11},{65,12},{89,18}};
  int vec_size = 20;
  elem heap[50];
  int heap_size = 0;

  elem v;
  v.id = 5;
  v.cost = 28;
  cout << "Inserting " << v << endl;
  insert(v, heap, heap_size);

  v.id = 9;
  v.cost = 13;
  cout << "Inserting " << v << endl;
  insert(v,heap,heap_size);

  v.id = 16;
  v.cost = 32;
  cout << "Inserting " << v << endl;
  insert(v,heap,heap_size);

  v.id = 3;
  v.cost = 7;
  cout << "Inserting " << v << endl;
  insert(v,heap,heap_size);

  v.id = 7;
  v.cost = 25;
  cout << "Inserting " << v << endl;
  insert(v,heap,heap_size);

  v.id = 20;
  v.cost = 27;
  cout << "Inserting " << v << endl;
  insert(v,heap,heap_size);

  v.id = 50;
  v.cost = 31;
  cout << "Inserting " << v << endl;
  insert(v,heap,heap_size);

  cout << "Heap: ";
  for(int i = 1; i <= heap_size; ++i)
  {
    cout << heap[i] << " ";
  }
  cout << "Heap size: " << heap_size << endl;
  int c = 3;
  cout << "min child of " << c << " "<< minChild(c, heap, heap_size) << endl;

  cout << "Pop heap: " << pop(heap,heap_size) << endl;

  cout << endl;
  for(int i = 1; i <= heap_size; ++i)
  {
    cout << heap[i] << " ";
  }
  cout << endl;


  cout << "Heapify vector\n";
  heapify(vec,vec_size);

  for(int i = 0; i <= vec_size; ++i)
  {
    cout << vec[i] << " ";
  }
  cout << endl << endl;

  while(vec_size > 0)
  {
    cout << "Pop Heap\n";
    cout << pop(vec,vec_size) << " Removed" << endl;

    for(int i = 0; i <= vec_size; ++i)
    {
      cout << vec[i] << " ";
    }
    cout << endl;

  }
  cout << endl;





}
