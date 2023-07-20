#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

int v_num = 0;
int e_num = 0;
int F0 = 0, F1 = 0, F2 = 0;

vector<vector<int>> edge_index;
vector<vector<float>> edge_val;
vector<int> degree;
vector<int> raw_graph;

float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter;

void readGraph(char *fname)
{
  ifstream infile(fname);

  int source;
  int end;

  infile >> v_num >> e_num;

  // raw_graph.resize(e_num * 2);

  while (!infile.eof())
  {
    infile >> source >> end;
    if (infile.peek() == EOF)
      break;
    raw_graph.push_back(source);
    raw_graph.push_back(end);
  }
}

void raw_graph_to_AdjacencyList()
{
  int src;
  int dst;

  edge_index.resize(v_num);
  edge_val.resize(v_num);
  degree.resize(v_num, 0);
  for (int i = 0; i < raw_graph.size() / 2; i++)
  {
    src = raw_graph[2 * i];
    dst = raw_graph[2 * i + 1];
    edge_index[dst].push_back(src);
    degree[src]++;
  }
}

void edgeNormalization()
{
#pragma omp parallel for
  for (int i = 0; i < v_num; i++)
  {
    for (int j = 0; j < edge_index[i].size(); j++)
    {
      float val = 1 / sqrt(degree[i]) / sqrt(degree[edge_index[i][j]]);
      edge_val[i].push_back(val);
    }
  }
}

void readFloat(char *fname, float *&dst, int num)
{
  dst = (float *)malloc(num * sizeof(float));
  FILE *fp = fopen(fname, "rb");
  fread(dst, num * sizeof(float), 1, fp);
  fclose(fp);
}

void initFloat(float *&dst, int num)
{
  dst = (float *)malloc(num * sizeof(float));
  memset(dst, 0, num * sizeof(float));
}

void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W)
{

  float(*tmp_in_X)[in_dim] = (float(*)[in_dim])in_X;
  float(*tmp_out_X)[out_dim] = (float(*)[out_dim])out_X;
  float(*tmp_W)[out_dim] = (float(*)[out_dim])W;

  int row_seg_size = 16;
  int col_seg_size = 4;
  int last_row = v_num & -4;  // 最后一个残缺块的第一行

  // float* X_after_seg = NULL;
  // float* W_after_seg = NULL;
  // vector<float> X_after_seg;
  // vector<float> W_after_seg;

  float X_after_seg[v_num * in_dim];
  float W_after_seg[in_dim * out_dim];

/*
  分块策略：X每16行分为1块。不足16行的部分以每4行为1块，不足4行的部分单独为1块。W每4列分为1块。
  存储顺序：X：块内按列存储，即先存块内的第1列、再存第2列...第in_dim列，然后再存下一个快。
           W：块内按行存储，即先存块内第1行、再存第2行...第in_dim列，再存下一个块。
  计算顺序：按照存储顺序计算，以达到连续访存的目的。
*/

  // 重排 X  以16行为一个单位来划分
  // #pragma omp parallel for
  // #pragma omp simd
  for (int i = 0; i < (v_num & -row_seg_size); i += row_seg_size) // i为每个块第一行
  {
    for (int j = 0; j < in_dim; j++) // j为列
    {
      for (int k = i; k < i + row_seg_size; k++) // 行
      {
        // *X_after_seg++ = tmp_in_X[k][j];
        // X_after_seg.push_back(tmp_in_X[k][j]);
        X_after_seg[i * in_dim + j * row_seg_size + k - i] = tmp_in_X[k][j];
      }
    }
  }
  // 重排 X  最后不足16行的部分，以4行划分
  // #pragma omp parallel for
  // #pragma omp simd
  for (int i = (v_num & -row_seg_size); i < v_num; i += 4) // i 为每个块的第一行
  {
    for (int j = 0; j < in_dim; j++)
    {
      for (int k = i; k < i + 4; k++)
      {
        // *X_after_seg++ = tmp_in_X[k][j];
        // X_after_seg.push_back(tmp_in_X[k][j]);
        X_after_seg[i * in_dim + j * 4 + k - i] = tmp_in_X[k][j];
      }
    }
    // last_row = i;
  }
  // 重排 X  最后不足4行的部分，直接存起来，不划分了
  // #pragma omp parallel for
  // #pragma omp simd
  for (int j = 0; j < in_dim; j++)
  {
    for (int i = last_row; i < v_num; i++)
    {
      // *X_after_seg++ = tmp_in_X[i][j];
      // X_after_seg.push_back(tmp_in_X[i][j]);
      X_after_seg[last_row * in_dim + j * (v_num % 4) + i - last_row] = tmp_in_X[i][j];
    }
  }

  // 重排 W  以4列为一个单位来划分
  // #pragma omp parallel for
  // #pragma omp simd
  for (int i = 0; i < out_dim; i += col_seg_size) // 块
  {
    for (int j = 0; j < in_dim; j++) // 行
    {
      for (int k = i; k < i + col_seg_size; k++) // 列
      {
        // *W_after_seg++ = tmp_W[j][k];
        // W_after_seg.push_back(tmp_W[j][k]);
        W_after_seg[i * in_dim + j * col_seg_size + k - i] = tmp_W[j][k];
      }
    }
  }

  // 开始计算
  // 计算划分成16行的部分
  // #pragma omp parallel for
  // #pragma omp simd
  for (int i = 0; i < v_num - row_seg_size + 1; i += row_seg_size)
  {
    for (int j = 0; j < out_dim - col_seg_size + 1; j += col_seg_size)
    {
      for (int k = 0; k < in_dim; k++)
      {
        for (int l = 0; l < row_seg_size; l++)
        {
          for (int m = 0; m < col_seg_size; m++)
          {
            tmp_out_X[i + l][j + m] += X_after_seg[i * in_dim + k * row_seg_size + l] * W_after_seg[j * in_dim + k * col_seg_size + m];
          }
        }
      }
    }
  }
  // 计算不足16行的部分，即以4行划分的那些
  // #pragma omp parallel for
  // #pragma omp simd
  for (int i = (v_num & -16); i < v_num - 3; i += 4)
  {
    for (int j = 0; j < out_dim - col_seg_size + 1; j += col_seg_size)
    {
      for (int k = 0; k < in_dim; k++)
      {
        for (int l = 0; l < 4; l++)
        {
          for (int m = 0; m < col_seg_size; m++)
          {
            tmp_out_X[i + l][j + m] += X_after_seg[i * in_dim + k * row_seg_size + l] * W_after_seg[j * in_dim + k * col_seg_size + m];
          }
        }
      }
    }
  }
  // 计算不足4行的部分，即直接存储的那些
  // #pragma omp parallel for
  // #pragma omp simd
  for (int j = 0; j < out_dim - col_seg_size + 1; j += col_seg_size)
  {
    for (int k = 0; k < in_dim; k++)
    {
      for (int l = 0; l < v_num % 4; l++)
      {
        for (int m = 0; m < col_seg_size; m++)
        {
          tmp_out_X[(v_num & -4) + l][j + m] += X_after_seg[(v_num & -4) * in_dim + k * (v_num % 4) + l] * W_after_seg[j * in_dim + k * col_seg_size + m];
        }
      }
    }
  }
}

void AX(int dim, float *in_X, float *out_X)
{
  float(*tmp_in_X)[dim] = (float(*)[dim])in_X;
  float(*tmp_out_X)[dim] = (float(*)[dim])out_X;
  // #pragma omp parallel for
  for (int i = 0; i < v_num; i++)
  {
    vector<int> &nlist = edge_index[i];
    for (int j = 0; j < nlist.size(); j++)
    {
      int nbr = nlist[j];
      for (int k = 0; k < dim; k++)
      {
        tmp_out_X[i][k] += tmp_in_X[nbr][k] * edge_val[i][j];
      }
    }
  }
}

void ReLU(int dim, float *X)
{
#pragma omp parallel for
  for (int i = 0; i < v_num * dim; i++)
    if (X[i] < 0)
      X[i] = 0;
}

void LogSoftmax(int dim, float *X)
{
  float(*tmp_X)[dim] = (float(*)[dim])X;
#pragma omp parallel for
  for (int i = 0; i < v_num; i++)
  {
    float max = tmp_X[i][0];
    for (int j = 1; j < dim; j++)
    {
      if (tmp_X[i][j] > max)
        max = tmp_X[i][j];
    }

    float sum = 0;
    for (int j = 0; j < dim; j++)
    {
      sum += exp(tmp_X[i][j] - max);
    }
    sum = log(sum);

    for (int j = 0; j < dim; j++)
    {
      tmp_X[i][j] = tmp_X[i][j] - max - sum;
    }
  }
}

float MaxRowSum(float *X, int dim)
{
  float(*tmp_X)[dim] = (float(*)[dim])X;
  float max = -__FLT_MAX__;

#pragma omp parallel for reduction(max : max)
  // #pragma omp parallel for
  for (int i = 0; i < v_num; i++)
  {
    float sum = 0;
    for (int j = 0; j < dim; j++)
    {
      sum += tmp_X[i][j];
    }
    if (sum > max)
      max = sum;
  }
  return max;
}

// float MaxRowSum(float *X, int dim) {
//     float(*tmp_X)[dim] = (float(*)[dim])X;
//     float max_sum = -__FLT_MAX__;
//     std::vector<float> row_sums(v_num, 0.0f);

//     // 计算每行的部分和
//     #pragma omp parallel for
//     for (int i = 0; i < v_num; i++) {
//         float sum = 0;
//         for (int j = 0; j < dim; j++) {
//             sum += tmp_X[i][j];
//         }
//         row_sums[i] = sum;
//     }

//     // 累加树算法求最大行和
//     for (int d = 1; d < v_num; d *= 2) {
//         #pragma omp parallel for
//         for (int i = 0; i < v_num; i += 2 * d) {
//             if (i + d < v_num) {
//                 row_sums[i] += row_sums[i + d];
//                 max_sum = std::max(max_sum, row_sums[i]);
//             }
//         }
//     }

//     return max_sum;
// }

void freeFloats()
{
  free(X0);
  free(W1);
  free(W2);
  free(X1);
  free(X2);
  free(X1_inter);
  free(X2_inter);
}

void somePreprocessing()
{
  // The graph  will be transformed into adjacency list, you can use other data
  // structure such as CSR
  raw_graph_to_AdjacencyList();
}

int main(int argc, char **argv)
{
  // Do NOT count the time of reading files, malloc, and memset
  F0 = atoi(argv[1]);
  F1 = atoi(argv[2]);
  F2 = atoi(argv[3]);

  readGraph(argv[4]);
  readFloat(argv[5], X0, v_num * F0);
  readFloat(argv[6], W1, F0 * F1);
  readFloat(argv[7], W2, F1 * F2);

  initFloat(X1, v_num * F1);
  initFloat(X1_inter, v_num * F1);
  initFloat(X2, v_num * F2);
  initFloat(X2_inter, v_num * F2);

  // Time point at the start of the computation
  TimePoint start = chrono::steady_clock::now();

  // Preprocessing time should be included

  TimePoint start_pre = chrono::steady_clock::now();
  somePreprocessing();
  TimePoint end_pre = chrono::steady_clock::now();
  chrono::duration<double> l_durationSec00 = end_pre - start_pre;
  double pre_timeMs = l_durationSec00.count() * 1e3;
  printf("Preprocessing:  %.8f\n", pre_timeMs);

  TimePoint start_norm = chrono::steady_clock::now();
  edgeNormalization();
  TimePoint end_norm = chrono::steady_clock::now();
  chrono::duration<double> l_durationSec0 = end_norm - start_norm;
  double norm_timeMs = l_durationSec0.count() * 1e3;
  printf("Edge Normalization:  %.8f\n", norm_timeMs);

  // printf("Layer1 XW\n");
  TimePoint start_XW_1 = chrono::steady_clock::now();
  XW(F0, F1, X0, X1_inter, W1);
  TimePoint end_XW_1 = chrono::steady_clock::now();
  chrono::duration<double> l_durationSec1 = end_XW_1 - start_XW_1;
  double XW_timeMs_1 = l_durationSec1.count() * 1e3;
  printf("XW1:  %.8f\n", XW_timeMs_1);

  // printf("Layer1 AX\n");
  TimePoint start_AX_1 = chrono::steady_clock::now();
  AX(F1, X1_inter, X1);
  TimePoint end_AX_1 = chrono::steady_clock::now();
  chrono::duration<double> l_durationSec2 = end_AX_1 - start_AX_1;
  double AX_timeMs_1 = l_durationSec2.count() * 1e3;
  printf("AX1:  %.8f\n", AX_timeMs_1);

  // printf("Layer1 ReLU\n");
  TimePoint start_relu = chrono::steady_clock::now();
  ReLU(F1, X1);
  TimePoint end_relu = chrono::steady_clock::now();
  chrono::duration<double> l_durationSec11 = end_relu - start_relu;
  double relu_timeMs = l_durationSec11.count() * 1e3;
  printf("Relu:  %.8f\n", relu_timeMs);

  // printf("Layer2 XW\n");
  TimePoint start_XW_2 = chrono::steady_clock::now();
  XW(F1, F2, X1, X2_inter, W2);
  TimePoint end_XW_2 = chrono::steady_clock::now();
  chrono::duration<double> l_durationSec3 = end_XW_2 - start_XW_2;
  double XW_timeMs_2 = l_durationSec3.count() * 1e3;
  printf("XW2:  %.8f\n", XW_timeMs_2);

  // printf("Layer2 AX\n");
  TimePoint start_AX_2 = chrono::steady_clock::now();
  AX(F2, X2_inter, X2);
  TimePoint end_AX_2 = chrono::steady_clock::now();
  chrono::duration<double> l_durationSec4 = end_AX_2 - start_AX_2;
  double AX_timeMs_2 = l_durationSec4.count() * 1e3;
  printf("AX2:  %.8f\n", AX_timeMs_2);

  // printf("Layer2 LogSoftmax\n");
  TimePoint start_log = chrono::steady_clock::now();
  LogSoftmax(F2, X2);
  TimePoint end_log = chrono::steady_clock::now();
  chrono::duration<double> l_durationSec22 = end_log - start_log;
  double log_timeMs = l_durationSec22.count() * 1e3;
  printf("LogSoftmax:  %.8f\n", log_timeMs);

  // You need to compute the max row sum for result verification
  TimePoint start_sum = chrono::steady_clock::now();
  float max_sum = MaxRowSum(X2, F2);
  TimePoint end_sum = chrono::steady_clock::now();
  chrono::duration<double> l_durationSec5 = end_sum - start_sum;
  double end_timeMs = l_durationSec5.count() * 1e3;
  printf("Max Row Sum:  %.8f\n", end_timeMs);

  // Time point at the end of the computation
  TimePoint end = chrono::steady_clock::now();
  chrono::duration<double> l_durationSec = end - start;
  double l_timeMs = l_durationSec.count() * 1e3;

  // Finally, the max row sum and the computing time
  // should be print to the terminal in the following format
  printf("%.8f\n", max_sum);
  printf("%.8f\n", l_timeMs);

  // Remember to free your allocated memory
  freeFloats();
}
