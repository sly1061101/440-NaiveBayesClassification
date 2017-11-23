//
//  main.cpp
//  NaiveBayesClassifier
//
//  Created by Liuyi Shi on 11/6/17.
//  Copyright © 2017 Liuyi Shi. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <dirent.h> //windows开发工具没有这个头文件
#include <unistd.h>
#include <string.h>
//Get list of file
int GetFileName(char file_list[][40]){
    DIR * dir;
    struct dirent * ptr;
    int i=0;
    dir = opendir("./unseg"); //打开一个目录
    while((ptr = readdir(dir)) != NULL) //循环读取目录数据
    {
        if(strlen(ptr->d_name) > 5){
            strcpy(file_list[i],ptr->d_name ); //存储到数组
            if ( ++i>=100 ) break;
        }
    }
    closedir(dir);//关闭目录指针
    return i;
}

#define WIN_SIZE 1
void seperate(){
    char filename[100][40];
    int num;
    num = GetFileName(filename);
    FILE *pf_data;
    FILE *pf_label;
    pf_data = fopen("yesno_unseg_data.txt", "w");
    pf_label = fopen("yesno_unseg_label.txt", "w");
    for(int i = 0; i<num; ++i){
        FILE *pf;
        char data[25][151];
        char currentfile[40] = "./unseg/";
        strcat(currentfile, filename[i]);
        pf = fopen( currentfile , "r");
        for(int j = 0; j < 25; ++j){
            fread(data[j], sizeof(char), 150+1, pf);
        }
        fclose(pf);
        char digit[8];
        for(int j = 0; j<8; ++j)
            digit[j] = filename[i][2*j];
        int highenegry[150] = {0};
        int index[8];
        for(int p = 0; p<25; ++p){
            for(int q = 0; q<150; ++q){
                if(data[p][q] == ' ')
                    for( int j = -WIN_SIZE; j<=WIN_SIZE; ++j)
                        if(q+j>=0 && q+j <150)
                            highenegry[q+j]++;
            }
        }
        int id = 0;
        for(int p=1; p<149; ++p){
            if(highenegry[p]>=highenegry[p-1] && highenegry[p]>highenegry[p+1] && highenegry[p]>=10 && ( (id>0 && p-index[id-1]>5) || id==0 ) )
                index[id++] = p;
        }
        for(int j=0; j<8; ++j){
            char line_data[11];
            char line_label[2];
            line_label[0] = digit[j];
            line_label[1] = '\n';
            fwrite(line_label, sizeof(char), 2, pf_label);
            for(int p = 0; p<25; ++p){
                for(int q = 0; q<10; ++q){
                    line_data[q] = data[p][ index[j]+q-4 ];
                }
                line_data[10] = '\n';
                fwrite(line_data, sizeof(char), 11, pf_data);
            }
            fwrite("\n", sizeof(char), 1, pf_data);
            fwrite("\n", sizeof(char), 1, pf_data);
            fwrite("\n", sizeof(char), 1, pf_data);
        }
    }
    fclose(pf_data);
    fclose(pf_label);
}

int main(){
    seperate();
    return 0;
}
