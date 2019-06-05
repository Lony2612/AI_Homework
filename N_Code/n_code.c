#include<string.h>
#include "defines.h"
#include "pointer.h"
#include "utils.h"
#include "astar.h"

int Astar(int ***a,int **start,int **target,int path[maxState],int length);

int main()
{
    FILE *fp;

    char str[32]; 
    if((fp=fopen("Npuzzle_in.txt","r"))==NULL) {
        printf("cannot open file/n"); 
        exit(1);
    } 

    // 获取矩阵的大小
    if(fgets(str,32,fp)!=NULL)
        n = atoi(str);
    
    // 为列表申请空间
    start_NUM = (int *) malloc((n*n) * sizeof(int));
    end_NUM = (int *) malloc((n*n) * sizeof(int));

    // 获取初始状态和目标状态
    int idx = 0;
    char *p;
    while(!feof(fp)) {
        if(fgets(str,32,fp)!=NULL)
        // printf("%s",str);
        p = strtok(str, " ");
        while(p)
        {   
            if (idx < n*n)
                start_NUM[idx] = atoi(p);
            else
            {
                end_NUM[idx-n*n] = atoi(p);
            }
            
            idx ++;
            p = strtok(NULL, " ");  
        }
    }
    fclose(fp);

    // 为初始矩阵和目标矩阵分配空间
    start = initial_Matrix(n);
    target = initial_Matrix(n);

    // 矩阵赋值
    for(int i= 0;i<n;i++)
        for(int j=0;j<n;j++)
        {
            start[i][j] = start_NUM[i*n+j];
            target[i][j] = end_NUM[i*n+j];
        }

    int ***a = CreateGrid(n,n,maxState);

    if(!(calDe(start,n)%2 == calDe(target,n)%2)){
        printf("no answer\n");
        return 0;
    }

    int path[maxState] = {0};
    int res =  Astar(a,start,target,path,n);
    if(res == -1){
        printf("reach the biggest state\n");
        return 0;
    }
    int shortest[maxState] = {0},j = 0;
    while(res != 0){
        shortest[j++] = res;
        res = path[res];
    }
    freopen("Npuzzle_out.txt","w",stdout);
    printf("total steps: %d\n",j);
    printf("initial state\n");
    show(a,0,n);

    for(int i = j - 1;i > 0;i --){
        printf("step %d\n",j-i);
        show(a,shortest[i],n);
    }

    printf("target state\n");
    show(a,shortest[0],n);
    freopen("CON","w",stdout);

    // 释放占用的内存
    free_pointer(start,n);
    free_pointer(target,n);
    FreeGrid(a,n,n,maxState);
    return 0;
}