#ifndef _UTILS_h_
#define _UTILS_h_
#include<stdlib.h>
#include<stdio.h>

bool isEqual(int ***a,int **b,int n, int len)
{
    for(int i = 0;i < len;i ++){
        for(int j = 0;j < len;j ++){
            if(a[i][j][n] != b[i][j]) return false;
        }
    }
    return true;
}

bool isEqual(int **a,int **b, int len)
{
    for(int i = 0;i < len;i ++){
        for(int j = 0;j < len;j ++){
            if(a[i][j] != b[i][j]) return false;
        }
    }
    return true;
}

int evalute(int **state,int **target, int len)
{
    int num = 0;
    for(int i = 0;i < len;i ++){
        for(int j = 0;j < len;j ++)
            if(state[i][j] != target[i][j]) num ++;
    }
    return num;
}

void findBrack(int **a,int x,int y, int len)
{
    for(int i = 0;i < len;i ++){
        for(int j = 0;j < len;j ++){
            if(a[i][j] == 0) {
                x = i;y = j;
                return;
            }
        }
    }
}

bool move(int **a,int **b,int dir, int len)
{
    //1 up 2 down 3 left 4 right
    int x = 0,y = 0;
    for(int i = 0;i < len;i ++){
        for(int j = 0;j < len;j ++){
            b[i][j] = a[i][j];
            if(a[i][j] == 0) {
                x = i;y = j;
            }
        }
    }
    if(x == 0 && dir == 1) return false;
    if(x == len-1 && dir == 2) return false;
    if(y == 0 && dir == 3) return false;
    if(y == len-1 && dir == 4) return false;
    if(dir == 1){b[x-1][y] = 0;b[x][y] = a[x-1][y];}
    else if(dir == 2){b[x+1][y] = 0;b[x][y] = a[x+1][y];}
    else if(dir == 3){b[x][y-1] = 0;b[x][y] = a[x][y-1];}
    else if(dir == 4){b[x][y+1] = 0;b[x][y] = a[x][y+1];}
    else return false;
    return true;
}

void statecpy(int ***a,int **b,int n, int len)
{
    for(int i = 0;i < len;i ++){
        for(int j = 0;j < len;j ++){
            a[i][j][n] = b[i][j];
        }
    }
}

void getState(int ***a,int **b,int n, int len)
{
    for(int i = 0;i < len;i ++){
        for(int j = 0;j < len;j ++){
            b[i][j] = a[i][j][n];
        }
    }
}

void statecpy(int **a,int **b, int len)
{
    for(int i = 0;i < len;i++){
        for(int j = 0;j < len;j++)
            a[i][j] = b[i][j];
    }
}

int checkAdd(int ***a,int **b,int n,int len)
{
    for(int i = 0;i < n;i ++){
        if(isEqual(a,b,i,len)) return i;
    }
    return -1;
}

void show(int ***a,int n, int len)
{
    for(int i = 0;i < len;i ++){
        for(int j =0;j < len;j ++){
            printf("%d ",a[i][j][n]);
        }
        printf("\n");
    }
}

int calDe(int **a,int len)
{
    int sum = 0;
    for(int i = 0;i < len*len;i ++){
        for(int j = i+1;j < len*len;j ++){
            int m,n,c,d;
            m = i/len;n = i%len;
            c = j/len;d = j%len;
            if(a[c][d] == 0) continue;
            if(a[m][n] > a[c][d]) sum ++;
        }
    }
    return sum;
}

#endif