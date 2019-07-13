#ifndef _POINTER_h_
#define _POINTER_h_
#include<stdlib.h>

void free_pointer(int** array, int len){
    for(int i=0;i<len;i++)
        free(array[i]);
    free(array);
}

int*** CreateGrid(int m,int n,int t)
{
    int i = 0;
    int k = 0;
    int*** tt = NULL; 
    if((m > 0) && (n > 0) && (t > 0))
    {
        tt = (int***)malloc(sizeof(int)*m);
        for(i = 0;i < m;i++)
        {
            tt[i] = (int**)malloc(sizeof(int)*n);
            for (k = 0;k < n;k++)
            {
                tt[i][k] = (int*)malloc(sizeof(int)*t);
            }
        }
    }
    return tt;
}

void FreeGrid(int*** tt,int m,int n,int t)
{
    int i = 0;
    int j = 0;
    if(tt != NULL)
    {
        for(i = 0;i < m;i++)
        {
            for (j = 0;j < n;j++)
            {
                free((tt[i][j]));
            }
            free(tt[i]);
        }
        free(tt);
        tt = NULL;
    }
}

int** initial_Matrix(int len){
    int **matrix = (int**)malloc(len*sizeof(int*));
    for (int i=0;i<len;i++)
        matrix[i] = (int*)malloc(len*sizeof(int));
    return matrix;
}

int** initial_Matrix_with_zero(int len){
    int **matrix = (int**)malloc(len*sizeof(int*));
    for (int i=0;i<len;i++)
        matrix[i] = (int*)malloc(len*sizeof(int));
    for(int i=0;i<len;i++)
    for(int j=0;j<len;j++)
        matrix[i][j] = 0;
    return matrix;
}

#endif