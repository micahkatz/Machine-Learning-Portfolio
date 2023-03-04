#include <math.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>

using namespace std;

double sensitivity(const std::vector<int>& actual,
                   const std::vector<int>& predicted) {
    int truePositive = 0;
    int falseNegative = 0;
    for (int i = 0; i < actual.size(); i++) {
        if (actual[i] == 1 && predicted[i] == 1) {
            truePositive++;
        } else if (actual[i] == 1 && predicted[i] == 0) {
            falseNegative++;
        }
    }
    return static_cast<double>(truePositive) / (truePositive + falseNegative);
}
double specificity(const std::vector<int>& actual,
                   const std::vector<int>& predicted) {
    int trueNegative = 0;
    int falsePositive = 0;
    for (int i = 0; i < actual.size(); i++) {
        if (actual[i] == 0 && predicted[i] == 0) {
            trueNegative++;
        } else if (actual[i] == 0 && predicted[i] == 1) {
            falsePositive++;
        }
    }
    return static_cast<double>(trueNegative) / (trueNegative + falsePositive);
}

double predictWithLogisticRegression(double x, double beta0, double beta1) {
    return 1.0 / (1.0 + exp(-(beta0 + beta1 * x)));
}

int main(int argc, char** argv) {
    ifstream inputCSV;
    string fileName = "titanic_project.csv";
    inputCSV.open(fileName);

    string currLine;
    string id, pclass, survived, sex, age;
    const int MAX_LEN = 2000;
    vector<string> id_vector(MAX_LEN);
    vector<int> pclass_vector(MAX_LEN);
    vector<int> survived_vector(MAX_LEN);
    vector<int> sex_vector(MAX_LEN);
    vector<double> age_vector(MAX_LEN);
    if (!inputCSV.is_open()) {
        cout << "Could not open file " << fileName << endl;
        return 1;  // return an error
    }

    // read csv data into vectors

    // read heading
    getline(inputCSV, currLine);
    int numObservations = 0;

    while (inputCSV.good()) {
        getline(inputCSV, id,
                ',');  // will get the first item and stop at the comma
        getline(inputCSV, pclass, ',');
        getline(inputCSV, survived, ',');
        getline(inputCSV, sex, ',');
        getline(inputCSV, age,
                '\n');  // will get the last item and stop at the newline

        id_vector.at(numObservations) = id;
        pclass_vector.at(numObservations) = stoi(pclass);
        survived_vector.at(numObservations) = stoi(survived);
        sex_vector.at(numObservations) = stoi(sex);
        age_vector.at(numObservations) = stod(age);

        numObservations++;
    }

    id_vector.resize(numObservations);
    pclass_vector.resize(numObservations);
    survived_vector.resize(numObservations);
    sex_vector.resize(numObservations);
    age_vector.resize(numObservations);

    inputCSV.close();  // Done with file, so close it

    cout << "Number of records: " << numObservations << endl;

    // predict SURVIVED based on SEX (first 800 rows for training)

    int numTrainingRows = 800;

    // divide into training data

    vector<string> id_training(id_vector.begin(),
                               id_vector.begin() + numTrainingRows);
    vector<int> pclass_training(pclass_vector.begin(),
                                pclass_vector.begin() + numTrainingRows);
    vector<int> survived_training(survived_vector.begin(),
                                  survived_vector.begin() + numTrainingRows);
    vector<int> sex_training(sex_vector.begin(),
                             sex_vector.begin() + numTrainingRows);
    vector<double> age_training(age_vector.begin(),
                                age_vector.begin() + numTrainingRows);

    // divide into testing data

    vector<string> id_testing(id_vector.begin() + numTrainingRows,
                              id_vector.end());
    vector<int> pclass_testing(pclass_vector.begin() + numTrainingRows,
                               pclass_vector.end());
    vector<int> survived_testing(survived_vector.begin() + numTrainingRows,
                                 survived_vector.end());
    vector<int> sex_testing(sex_vector.begin() + numTrainingRows,
                            sex_vector.end());
    vector<double> age_testing(age_vector.begin() + numTrainingRows,
                               age_vector.end());

    // initialize our betas to zero

    double beta0 = 0.0;
    double beta1 = 0.0;

    int numTrainingIterations = 5000;
    double trainingLearningRate = 0.01;  // learning rate for gradient descent

    // begin training
    chrono::high_resolution_clock::time_point trainingStart =
        chrono::high_resolution_clock::now();

    for (int i = 0; i < numTrainingIterations; i++) {
        double sumErrors = 0.0;

        // loop through training rows
        for (int j = 0; j < numTrainingRows; j++) {
            // predict SURVIVED based on SEX training data
            double survivalPrediction =
                predictWithLogisticRegression(sex_training[j], beta0, beta1);
            double error = survived_training[j] - survivalPrediction;
            sumErrors += error;

            // increment betas

            beta0 += trainingLearningRate * error * survivalPrediction *
                     (1.0 - survivalPrediction);
            beta1 += trainingLearningRate * error * sex_training[j] *
                     survivalPrediction * (1.0 - survivalPrediction);
        }
        // cout << "Iteration " << i << ", error = " << sumErrors << endl;
    }
    // end training
    chrono::high_resolution_clock::time_point trainingEnd =
        chrono::high_resolution_clock::now();

    chrono::high_resolution_clock::duration trainingTime =
        chrono::duration_cast<chrono::milliseconds>(trainingEnd -
                                                    trainingStart);
    cout << "Training time: " << trainingTime.count() << " milliseconds"
         << endl;

    int numTestingRows = numObservations - numTrainingRows;
    vector<int> predicted_survived(numTestingRows);
    for (int i = 0; i < numTestingRows; i++) {
        // predict SURVIVED based on SEX test data
        double survivalPrediction =
            predictWithLogisticRegression(sex_testing[i], beta0, beta1);
        predicted_survived[i] = (survivalPrediction >= 0.5) ? 1 : 0;
        // cout << "Predicted Survived " << i
        //      << ", value = " << predicted_survived[i] << endl;
    }

    // evaluate the accuracy

    int numCorrect = 0;
    for (int i = 0; i < numTestingRows; i++) {
        if (predicted_survived[i] == survived_testing[i]) {
            numCorrect++;
        }
    }
    double accuracy = static_cast<double>(numCorrect) / numTestingRows;

    cout << "Coefficients:" << endl;
    cout << "beta0 = " << beta0 << endl;
    cout << "beta1 = " << beta1 << endl;

    cout << "Accuracy = " << accuracy << endl;

    cout << "Specificity = "
         << specificity(survived_testing, predicted_survived) << endl;
    cout << "Sensitivity = "
         << sensitivity(survived_testing, predicted_survived) << endl;

    return 0;
}
