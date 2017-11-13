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

#define k 0.1
#define v 5
#define row_size 13
#define col_size 30
#define num_class 5

int main(){
    char dummy[row_size+1] = {0};
    char data[col_size][row_size+1] = {0};
    char label[2] = {0};
    int raw_data[num_class][row_size*col_size+1] = {0};
    
    FILE *pf_data;
    FILE *pf_label;
    pf_data = fopen("training_data.txt", "r");
    pf_label = fopen("training_labels.txt", "r");
    
    while( fread(label, sizeof(char), 2, pf_label) != 0 ){
        for(int i = 0; i < col_size; ++i){
            fread(data[i], sizeof(char), row_size + 1, pf_data);
        }
        fread(dummy, sizeof(char), 1, pf_data);
        fread(dummy, sizeof(char), 1, pf_data);
        fread(dummy, sizeof(char), 1, pf_data);
        int num;
        num = label[0] - '0';
        raw_data[num][0]++;
        for(int i = 0; i < col_size; ++i){
            for(int j = 0; j < row_size; ++j){
                if( data[i][j] != ' ' ){
                    raw_data[num][i * row_size + j + 1]++;
                }
            }
        }
    }
    fclose(pf_data);
    fclose(pf_label);
    
    //calculate p(class) and p(feature == 1 | class)
    float total_class = 0;

    for(int i = 0; i < num_class; ++i){
        total_class += raw_data[i][0];
    }
//    for(int i = 0; i < 10; ++i){
//        final_data[i][0] = raw_data[i][0] / total_class;
//        for(int j = 1; j < 785; ++j){
//            if(raw_data[i][j] != 0)
//                final_data[i][j] = raw_data[i][j] / total_class;
//            else
//                final_data[i][j] = k / (total_class + 2*k);
//        }
//    }
    
    //calculate the posteriors and make the decision
    float posterior[10];
    
    int total = 0;
    int error = 0;
    int total_digit[num_class] = {0};
    int error_digit[num_class] = {0};
    int confusion[num_class][num_class] = {0};
    
    FILE *pf_test_data;
    FILE *pf_test_label;
    pf_test_data = fopen("testing_data.txt", "r");
    pf_test_label = fopen("testing_labels.txt", "r");
    while( fread(label, sizeof(char), 2, pf_test_label) != 0 ){
        total++;
        int num;
        num = label[0] - '0';
        total_digit[num]++;
        
        for(int i = 0; i < col_size; ++i){
            fread(data[i], sizeof(char), row_size+1, pf_test_data);
        }
        fread(dummy, sizeof(char), 1, pf_data);
        fread(dummy, sizeof(char), 1, pf_data);
        fread(dummy, sizeof(char), 1, pf_data);
        
        float max = -99999999999999;
        int num_dec = 0;
        for(int h = 0; h < num_class; h++){
            posterior[h] = log( raw_data[h][0] / total_class );//final_data[h][0];
            for(int i = 0; i < col_size; ++i){
                for(int j = 0; j < row_size; ++j){
                    if( data[i][j] != ' ' ){
                        posterior[h] += log( (1.0*raw_data[h][i*row_size + j] + k)/(raw_data[h][0] + v*k) );
                    }
                    else{
                        posterior[h] += log( (1.0*raw_data[h][0] - raw_data[h][i*row_size + j] + k)/(raw_data[h][0] + v*k) );
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
    printf("OverAll:%f\n", 1- error/(float)total);
    for(int i = 0; i < num_class; ++i){
        printf("%d:%f\n", i, 1- error_digit[i]/(float)total_digit[i]);
    }
    for(int i = 0; i < num_class; ++i){
        for(int j = 0; j < num_class; ++j){
            printf("%f ", confusion[i][j]/(float)total_digit[i]);
        }
        printf("\n");
    }
    
    
    fclose(pf_test_data);
    fclose(pf_test_label);
    
}
