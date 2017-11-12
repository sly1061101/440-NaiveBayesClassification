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
#define v 2

int main(){
    char data[28][29] = {0};
    char label[2] = {0};
    int raw_data[10][785] = {0};
    float final_data[10][785] = {0};
    
    int it = 0;
    
    FILE *pf_data;
    FILE *pf_label;
    pf_data = fopen("trainingimages", "r");
    pf_label = fopen("traininglabels", "r");
    
    while( fread(label, sizeof(char), 2, pf_label) != 0 ){
        it++;
        for(int i = 0; i < 28; ++i){
            fread(data[i], sizeof(char), 29, pf_data);
        }
        int num;
        num = label[0] - '0';
        raw_data[num][0]++;
        for(int i = 0; i < 28; ++i){
            for(int j = 0; j < 28; ++j){
                if( (data[i][j] == '#') || (data[i][j] == '+') ){
                    raw_data[num][i * 28 + j + 1]++;
                }
            }
        }
    }
    fclose(pf_data);
    fclose(pf_label);
    
    //calculate p(class) and p(feature == 1 | class)
    float total_class = 0;

    for(int i = 0; i < 10; ++i){
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
    int total_digit[10] = {0};
    int error_digit[10] = {0};
    int confusion[10][10] = {0};
    
    FILE *pf_test_data;
    FILE *pf_test_label;
    pf_test_data = fopen("testimages", "r");
    pf_test_label = fopen("testlabels", "r");
    while( fread(label, sizeof(char), 2, pf_test_label) != 0 ){
        total++;
        int num;
        num = label[0] - '0';
        total_digit[num]++;
        
        for(int i = 0; i < 28; ++i){
            fread(data[i], sizeof(char), 29, pf_test_data);
        }
        
        float max = -99999999999999;
        int num_dec = 0;
        for(int h = 0; h < 10; h++){
            posterior[h] = log( raw_data[h][0] / total_class );//final_data[h][0];
            for(int i = 0; i < 28; ++i){
                for(int j = 0; j < 28; ++j){
                    if( (data[i][j] == '#') || (data[i][j] == '+') ){
                        posterior[h] += log( (1.0*raw_data[h][i*28 + j] + k)/(raw_data[h][0] + v*k) );
                    }
                    else{
                        posterior[h] += log( (1.0*raw_data[h][0] - raw_data[h][i*28 + j] + k)/(raw_data[h][0] + v*k) );
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
    for(int i = 0; i < 10; ++i){
        printf("%d:%f\n", i, 1- error_digit[i]/(float)total_digit[i]);
    }
    for(int i = 0; i < 10; ++i){
        for(int j = 0; j < 10; ++j){
            printf("%f ", confusion[i][j]/(float)total_digit[i]);
        }
        printf("\n");
    }
    
    
    fclose(pf_test_data);
    fclose(pf_test_label);
    
}



