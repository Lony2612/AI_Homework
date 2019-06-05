#ifndef _ASTAR_h_
#define _ASTAR_h_

#include "defines.h"
#include "pointer.h"
#include "utils.h"

int Astar(int ***a,int **start,int **target,int path[maxState],int length)
{
    bool visited[maxState] = {false};
    int fitness[maxState] = {0};
    int passLen[maxState] = {0};
    int** curpos = initial_Matrix(length);
    statecpy(curpos,start,length);
    int id = 0,Curid = 0;
    fitness[id] = evalute(curpos,target,length);
    statecpy(a,start,id++,length);
    while(!isEqual(curpos,target,length)){
        for(int i = 1;i < 5;i ++){//向四周找方向
            int** tmp = initial_Matrix_with_zero(length);
            if(move(curpos,tmp,i,length)){
                int state = checkAdd(a,tmp,id,length);
                if(state == -1){//not add
                    path[id] = Curid;
                    passLen[id] = passLen[Curid] + 1;
                    fitness[id] = evalute(tmp,target,length) + passLen[id];
                    statecpy(a,tmp,id++,length);
                }else{//add

                    int len = passLen[Curid] + 1,fit = evalute(tmp,target,length) + len;
                    if(fit < fitness[state]){
                        path[state] = Curid;
                        passLen[state] = len;
                        fitness[state] = fit;
                        visited[state] = false;
                    }
                }
            free_pointer(tmp,length);
            }
        }
        visited[Curid] = true;
        //找到适应度最小的最为下一个带搜索节点
        int minCur = -1;
        for(int i = 0;i < id;i ++)
            if(!visited[i] && (minCur == -1 || fitness[i] < fitness[minCur])) minCur = i;
        Curid = minCur;
        getState(a,curpos,Curid,length);
        if(id == maxState) return -1;
    }
    free_pointer(curpos,length);
    return Curid;
}


#endif