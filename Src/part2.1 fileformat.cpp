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

#define k 0.1
#define v 11
#define row_size 10
#define col_size 25
#define num_class 2

int main(){
    char dummy[row_size+1] = {0};
    char data[col_size][row_size+1] = {0};
    //char label[2] = {0};
    std::vector<std::vector<std::vector<int>>> training_data;

    training_data.resize(num_class);
    for(auto &i:training_data){
        i.resize( col_size + 1 );
        i[0].resize(1);
        for(int j = 1; j< col_size + 1; ++j)
            i[j].resize(row_size + 1);
    }
    
    FILE *pf_data;
    //FILE *pf_label;
    pf_data = fopen("yes_train.txt", "r");
    //pf_label = fopen("yesno_train_label", "r");
    
    while( fread(data[0], sizeof(char), row_size + 1, pf_data) != 0 ){
        for(int i = 1; i < col_size; ++i){
            fread(data[i], sizeof(char), row_size + 1, pf_data);
        }
        fread(dummy, sizeof(char), 1, pf_data);
        fread(dummy, sizeof(char), 1, pf_data);
        fread(dummy, sizeof(char), 1, pf_data);
        int num;
        num = 1;
        training_data[num][0][0]++;
        for(int i = 0; i<col_size; ++i){
            int num_high = 0;
            for(int j = 0; j<row_size; ++j)
                if(data[i][j] == ' ')
                    num_high++;
            training_data[num][i+1][num_high]++;
        }
    }
    fclose(pf_data);
    //fclose(pf_label);
    
    pf_data = fopen("no_train.txt", "r");
    //pf_label = fopen("yesno_train_label", "r");
    
    while( fread(data[0], sizeof(char), row_size + 1, pf_data) != 0 ){
        for(int i = 1; i < col_size; ++i){
            fread(data[i], sizeof(char), row_size + 1, pf_data);
        }
        fread(dummy, sizeof(char), 1, pf_data);
        fread(dummy, sizeof(char), 1, pf_data);
        fread(dummy, sizeof(char), 1, pf_data);
        int num;
        num = 0;
        training_data[num][0][0]++;
        for(int i = 0; i<col_size; ++i){
            int num_high = 0;
            for(int j = 0; j<row_size; ++j)
                if(data[i][j] == ' ')
                    num_high++;
            training_data[num][i+1][num_high]++;
        }
    }
    fclose(pf_data);
    //fclose(pf_label);
    
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
    
    FILE *pf_test_data;
    //FILE *pf_test_label;
    pf_test_data = fopen("yes_test.txt", "r");
    //pf_test_label = fopen("yesno_test_label.txt", "r");
    while( fread(data[0], sizeof(char), row_size+1, pf_test_data) != 0 ){
        total++;
        int num;
        num = 1;
        total_digit[num]++;
        
        for(int i = 1; i < col_size; ++i){
            fread(data[i], sizeof(char), row_size+1, pf_test_data);
        }
        fread(dummy, sizeof(char), 1, pf_data);
        fread(dummy, sizeof(char), 1, pf_data);
        fread(dummy, sizeof(char), 1, pf_data);
        
        float max = -99999999999999;
        int num_dec = 0;
        for(int h = 0; h < num_class; h++){
            posterior[h] = log( 1.0*training_data[h][0][0] / total_class );
            for(int i = 0; i<col_size; ++i){
                int num_high = 0;
                for(int j = 0; j<row_size; ++j)
                    if(data[i][j] == ' ')
                        num_high++;
                posterior[h] +=log( (1.0*training_data[h][i+1][num_high] + k) / (training_data[h][0][0] + k*v) );
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
    fclose(pf_test_data);
    //fclose(pf_test_label);
    
    pf_test_data = fopen("no_test.txt", "r");
    //pf_test_label = fopen("yesno_test_label.txt", "r");
    while( fread(data[0], sizeof(char), row_size+1, pf_test_data) != 0 ){
        total++;
        int num;
        num = 0;
        total_digit[num]++;
        
        for(int i = 1; i < col_size; ++i){
            fread(data[i], sizeof(char), row_size+1, pf_test_data);
        }
        fread(dummy, sizeof(char), 1, pf_data);
        fread(dummy, sizeof(char), 1, pf_data);
        fread(dummy, sizeof(char), 1, pf_data);
        
        float max = -99999999999999;
        int num_dec = 0;
        for(int h = 0; h < num_class; h++){
            posterior[h] = log( 1.0*training_data[h][0][0] / total_class );
            for(int i = 0; i<col_size; ++i){
                int num_high = 0;
                for(int j = 0; j<row_size; ++j)
                    if(data[i][j] == ' ')
                        num_high++;
                posterior[h] +=log( (1.0*training_data[h][i+1][num_high] + k) / (training_data[h][0][0] + k*v) );
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
    fclose(pf_test_data);
    //fclose(pf_test_label);
    
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
    return 0;
}

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
