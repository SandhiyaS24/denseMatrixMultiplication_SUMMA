# SUMMA-based Distributed Matrix Multiplication (GEMM)
 
A distributed dense matrix-matrix multiplication (GEMM) implementation using two variants of the **SUMMA (Scalable Universal Matrix Multiplication Algorithm)** protocol over MPI on a p × p process grid.
 
---
 
## Overview
 
This project implements two variants of distributed GEMM that compute `C = A × B` across a 2D MPI process grid:
 
- **Stationary-C SUMMA** — matrix C stays fixed; A and B blocks are broadcast across rows and columns each iteration.
- **Stationary-A SUMMA** — matrix A stays fixed; B blocks are broadcast column-wise, and C is assembled via reduce-scatter.
 
Both variants support square and rectangular (tall-and-skinny) matrices.
 
---
 
## Algorithms
 
### Stationary-C SUMMA
- Each process holds a fixed block of C, initialized to zero.
- For each step `k = 0 to p-1`:
  - Broadcast `A_ik` across process row `i`
  - Broadcast `B_kj` down process column `j`
  - Local update: `C_ij += A_temp × B_temp`
 
### Stationary-A SUMMA
- Each process holds a fixed block of A.
- For each step `j = 0 to p-1`:
  - Broadcast `B_kj` down process column `j`
  - Local update: `C_temp += A_ij × B_temp`
- Final C blocks assembled via **reduce-scatter** within each row.
 
---

## Build & Run
 
```bash
# Compile
make
 
# Run
make run
 
# Clean build artifacts
make clean
```
 
---
 
## Key MPI Operations Used
 
| Purpose | MPI Call |
|---|---|
| Process grid setup | `MPI_Cart_create`, `MPI_Cart_coords` |
| Row/column communicators | `MPI_Comm_split` |
| Data distribution | `MPI_Scatter` |
| Block broadcast | `MPI_Bcast` |
| Result collection | `MPI_Gather`, `MPI_Reduce_scatter` |
| Synchronization | `MPI_Barrier` |
 
---
 
## Test Cases
 
### Square Matrices (A: N×N, B: N×N → C: N×N)
 
| Size | Category |
|---|---|
| N = 4,096 / 8,192 | Small |
| N = 16,384 / 32,768 | Medium |
| N = 65,536 | Large |
 
### Rectangular — Tall-and-Skinny (A: M×K, B: K×N → C: M×N, K=128)
 
| M | Category |
|---|---|
| 4,096 / 8,192 | Small |
| 16,384 / 32,768 | Medium |
| 65,536 | Large |
 
