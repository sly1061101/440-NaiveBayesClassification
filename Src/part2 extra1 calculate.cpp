//
//  main.cpp
//  NaiveBayesClassifier
//
//  Created by Liuyi Shi on 11/6/17.
//  Copyright © 2017 Liuyi Shi. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <vector>

#define n 1
#define m 1
#define IsDisjoint true

#define k 0.02
#define v pow(2,m*n)
#define row_size 10
#define col_size 25
#define num_class 2

int GetFileName(char *path, char file_list[][40]);

int main(){
    char dummy[row_size+1] = {0};
    char data[col_size][row_size+1] = {0};
    char label[2] = {0};
    std::vector<std::vector<std::vector<int>>> training_data;
    char patch[n][m] = {0};
    
    training_data.resize(10);
    if( IsDisjoint ){
        for(auto &i:training_data){
            i.resize( (row_size/m) * (col_size/n) + 1 );
            i[0].resize(1);
            for(int j = 1; j<(row_size/m) * (col_size/n) + 1; ++j)
                i[j].resize(pow(2,m*n));
        }
    }
    else{
        for(auto &i:training_data){
            i.resize( (col_size-n+1) * (row_size-m+1) + 1 );
            i[0].resize(1);
            for(int j = 1; j<(col_size-n+1) * (row_size-m+1) + 1; ++j)
                i[j].resize(pow(2,m*n));
        }
    }
    
    FILE *pf_data;
    FILE *pf_label;
    pf_data = fopen("yesno_unseg_data.txt", "r");
    pf_label = fopen("yesno_unseg_label.txt", "r");
    
    while( fread(label, sizeof(char), 2, pf_label) != 0 ){
        for(int i = 0; i < col_size; ++i){
            fread(data[i], sizeof(char), row_size + 1, pf_data);
        }
        fread(dummy, sizeof(char), 1, pf_data);
        fread(dummy, sizeof(char), 1, pf_data);
        fread(dummy, sizeof(char), 1, pf_data);
        int num;
        num = label[0] - '0';
        training_data[num][0][0]++;
        if(IsDisjoint){
            for(int i = 0; i < col_size/n; ++i){
                for(int j = 0; j < row_size/m; ++j){
                    int hashVal=0;
                    for(int p = 0; p<n; ++p){
                        for(int q=0; q<m; ++q){
                            patch[p][q] = data[i*n+p][j*m+q];
                            if(patch[p][q] != ' '){
                                hashVal += pow(2,m*p+q);
                            }
                        }
                    }
                    training_data[num][(row_size/m)*i+j+1][hashVal]++;
                }
            }
        }
        else{
            for(int i = 0; i < col_size-n+1; ++i){
                for(int j = 0; j < row_size-m+1; ++j){
                    int hashVal=0;
                    for(int p = 0; p<n; ++p){
                        for(int q=0; q<m; ++q){
                            patch[p][q] = data[i+p][j+q];
                            if(patch[p][q] != ' '){
                                hashVal += pow(2,m*p+q);
                            }
                        }
                    }
                    training_data[num][(row_size-m+1)*i+j+1][hashVal]++;
                }
            }
        }
        
    }
    fclose(pf_data);
    fclose(pf_label);
    
    float total_class = 0;
    
    for(int i = 0; i < num_class; ++i){
        total_class += training_data[i][0][0];
    }
    
    //calculate the posteriors and make the decision
    float posterior[num_class] = {0};
    
    int total = 0;
    int error = 0;
    int total_digit[num_class] = {0};
    int error_digit[num_class] = {0};
    int confusion[num_class][num_class] = {0};
    
    char filename[100][40];
    int num_file;
    num_file = GetFileName("./yesno_unsegmented/yes_test/",filename);
    
    for(int r = 0; r<num_file; ++r){
        FILE *pf;
        char currentfile[40] = "./yesno_unsegmented/yes_test/";
        strcat(currentfile, filename[r]);
        pf = fopen( currentfile , "r");
        for(int j = 0; j < col_size; ++j){
            fread(data[j], sizeof(char), row_size+1, pf);
        }
        
        total++;
        int num = 1;
        
        total_digit[num]++;
        
        float max = -99999999999999;
        int num_dec = 0;
        for(int h = 0; h < num_class; h++){
            posterior[h] = log( 1.0*training_data[h][0][0] / total_class );
            if(IsDisjoint){
                for(int i = 0; i < col_size/n; ++i){
                    for(int j = 0; j < row_size/m; ++j){
                        int hashVal=0;
                        for(int p = 0; p<n; ++p){
                            for(int q=0; q<m; ++q){
                                patch[p][q] = data[i*n+p][j*m+q];
                                if(patch[p][q] != ' '){
                                    hashVal += pow(2,m*p+q);
                                }
                            }
                        }
                        posterior[h] +=log( (1.0*training_data[h][(row_size/m)*i+j+1][hashVal] + k) / (training_data[h][0][0] + k*v) );
                    }
                }
            }
            else{
                for(int i = 0; i < col_size-n+1; ++i){
                    for(int j = 0; j < row_size-m+1; ++j){
                        int hashVal=0;
                        for(int p = 0; p<n; ++p){
                            for(int q=0; q<m; ++q){
                                patch[p][q] = data[i+p][j+q];
                                if(patch[p][q] != ' '){
                                    hashVal += pow(2,m*p+q);
                                }
                            }
                        }
                        posterior[h] +=log( (1.0*training_data[h][(row_size-m+1)*i+j+1][hashVal] + k) / (training_data[h][0][0] + k*v) );
                    }
                }
            }
            if(posterior[h] > max){
                max = posterior[h];
                num_dec = h;
            }
        }
        confusion[num][num_dec]++;
        if(num != num_dec){
            error++;
            error_digit[num]++;
        }
        
    }
    
    num_file = GetFileName("./yesno_unsegmented/no_test/",filename);
    
    for(int r = 0; r<num_file; ++r){
        FILE *pf;
        char currentfile[40] = "./yesno_unsegmented/no_test/";
        strcat(currentfile, filename[r]);
        pf = fopen( currentfile , "r");
        for(int j = 0; j < col_size; ++j){
            fread(data[j], sizeof(char), row_size+1, pf);
        }
        
        total++;
        int num = 0;
        total_digit[num]++;
        
        float max = -99999999999999;
        int num_dec = 0;
        for(int h = 0; h < num_class; h++){
            posterior[h] = log( 1.0*training_data[h][0][0] / total_class );
            if(IsDisjoint){
                for(int i = 0; i < col_size/n; ++i){
                    for(int j = 0; j < row_size/m; ++j){
                        int hashVal=0;
                        for(int p = 0; p<n; ++p){
                            for(int q=0; q<m; ++q){
                                patch[p][q] = data[i*n+p][j*m+q];
                                if(patch[p][q] != ' '){
                                    hashVal += pow(2,m*p+q);
                                }
                            }
                        }
                        posterior[h] +=log( (1.0*training_data[h][(row_size/m)*i+j+1][hashVal] + k) / (training_data[h][0][0] + k*v) );
                    }
                }
            }
            else{
                for(int i = 0; i < col_size-n+1; ++i){
                    for(int j = 0; j < row_size-m+1; ++j){
                        int hashVal=0;
                        for(int p = 0; p<n; ++p){
                            for(int q=0; q<m; ++q){
                                patch[p][q] = data[i+p][j+q];
                                if(patch[p][q] != ' '){
                                    hashVal += pow(2,m*p+q);
                                }
                            }
                        }
                        posterior[h] +=log( (1.0*training_data[h][(row_size-m+1)*i+j+1][hashVal] + k) / (training_data[h][0][0] + k*v) );
                    }
                }
            }
            if(posterior[h] > max){
                max = posterior[h];
                num_dec = h;
            }
        }
        confusion[num][num_dec]++;
        if(num != num_dec){
            error++;
            error_digit[num]++;
        }
        
    }

    printf("OverAll:\n%f\n", 1- error/(float)total);
    printf("Accuracy for each digit:\n");
    for(int i = 0; i < num_class; ++i){
        printf("%d:%f\n", i, 1- error_digit[i]/(float)total_digit[i]);
    }
    printf("Confusion Matrix:\n");
    for(int i = 0; i < num_class; ++i){
        for(int j = 0; j < num_class; ++j){
            printf("%f ", confusion[i][j]/(float)total_digit[i]);
        }
        printf("\n");
    }
    
}

#include <sys/types.h>
#include <dirent.h> //windows开发工具没有这个头文件
#include <unistd.h>
#include <string.h>
//Get list of file
int GetFileName(char *path, char file_list[][40]){
    DIR * dir;
    struct dirent * ptr;
    int i=0;
    dir = opendir(path); //打开一个目录
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


