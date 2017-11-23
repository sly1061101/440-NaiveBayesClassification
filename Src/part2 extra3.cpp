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
