#include <math.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <tuple>
#include <vector>

using namespace std;

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
        age_vector.at(numObservations) = round(stod(age) * 10.0) / 10.0;

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

    // begin training
    chrono::high_resolution_clock::time_point trainingStart =
        chrono::high_resolution_clock::now();

    // calculate prior probabilities

    int totalNumSurvivors = 0;
    int numNonSurvivors = 0;
    int totalObservations = 0;

    for (int i = 0; i < numTrainingRows; i++) {
        if (survived_training[i] == 1) {
            totalNumSurvivors++;
        } else {
            numNonSurvivors++;
        }
        totalObservations++;
    }

    double priorSurvival =
        static_cast<double>(totalNumSurvivors) / totalObservations;
    double priorNonSurvival =
        static_cast<double>(numNonSurvivors) / totalObservations;

    // calculate p (Age | Survival)

    // count the number of survivors in each age group
    // {age: totalNumSurvivors}

    map<double, double> numSurvivorsGivenAge;
    map<double, double> numNonSurvivorsGivenAge;
    for (int i = 0; i < age_training.size(); i++) {
        double currAge = age_training.at(i);
        if (numSurvivorsGivenAge.find(currAge) == numSurvivorsGivenAge.end()) {
            // the age has not been stored yet
            if (survived_training[i] == 1) {  // survived
                numSurvivorsGivenAge.insert(pair<double, double>(currAge, 1));
            } else {  // did not survive
                numNonSurvivorsGivenAge.insert(
                    pair<double, double>(currAge, 1));
            }
        } else {                              // age has already been added
            if (survived_training[i] == 1) {  // survived
                map<double, double>::iterator foundAgeIterator =
                    numSurvivorsGivenAge.find(currAge);
                if (foundAgeIterator != numSurvivorsGivenAge.end()) {
                    foundAgeIterator->second += 1;
                }
            } else {  // did not survive
                map<double, double>::iterator foundAgeIterator =
                    numNonSurvivorsGivenAge.find(currAge);
                if (foundAgeIterator != numNonSurvivorsGivenAge.end()) {
                    foundAgeIterator->second += 1;
                } else {
                    numNonSurvivorsGivenAge.insert(
                        pair<double, double>(currAge, 1));
                }
            }
        }
    }

    // for (map<double, double>::iterator it = numSurvivorsGivenAge.begin();
    //      it != numSurvivorsGivenAge.end(); ++it) {
    //     double age = it->first;
    //     double numSurvivors = it->second;
    //     double numNonSurvivors = 0;
    //     if (numNonSurvivorsGivenAge.find(age) !=
    //         numNonSurvivorsGivenAge.end()) {
    //         numNonSurvivors = numNonSurvivorsGivenAge.find(age)->second;
    //     }
    //     double totalNumObservationsForAge =
    //         count(age_training.begin(), age_training.end(), age);
    //     double pAgeGivenSurvival = numSurvivors / totalNumObservationsForAge;
    //     double pAgeGivenNonSurvival =
    //         numNonSurvivors / totalNumObservationsForAge;
    //     cout << "Age: " << age << ", Num Survived: " << numSurvivors
    //          << ", Num Did Not Survive: " << numNonSurvivors
    //          << ", P(Age | Survival): " << pAgeGivenSurvival
    //          << ", P(Age | Non-Survival): " << pAgeGivenNonSurvival
    //          << std::endl;
    // }

    // calculate p (pclass | Survival)

    map<int, double> numSurvivorsGivenPclass;
    map<int, double> numNonSurvivorsGivenPclass;
    for (int i = 0; i < pclass_training.size(); i++) {
        int currPclass = pclass_training.at(i);
        if (numSurvivorsGivenPclass.find(currPclass) ==
            numSurvivorsGivenPclass
                .end()) {  // the pclass has not been stored yet
            if (survived_training[i] == 1) {  // survived
                numSurvivorsGivenPclass.insert(
                    pair<int, double>(currPclass, 1));
            } else {
                numNonSurvivorsGivenPclass.insert(
                    pair<int, double>(currPclass, 1));
            }
        } else {                              // pclass has already been added
            if (survived_training[i] == 1) {  // survived
                map<int, double>::iterator foundPclassIterator =
                    numSurvivorsGivenPclass.find(currPclass);
                if (foundPclassIterator != numSurvivorsGivenPclass.end()) {
                    foundPclassIterator->second += 1;
                }
            } else {  // non-survived
                map<int, double>::iterator foundPclassIterator =
                    numNonSurvivorsGivenPclass.find(currPclass);
                if (foundPclassIterator != numNonSurvivorsGivenPclass.end()) {
                    foundPclassIterator->second += 1;
                }
            }
        }
    }

    // for (map<int, double>::iterator it = numSurvivorsGivenPclass.begin();
    //      it != numSurvivorsGivenPclass.end(); ++it) {
    //     int pclass = it->first;
    //     double numSurvivors = it->second;
    //     double totalNumObservationsForPclass =
    //         count(pclass_training.begin(), pclass_training.end(), pclass);
    //     double pPclassGivenSurvival =
    //         numSurvivors / totalNumObservationsForPclass;
    //     cout << "Pclass: " << pclass << ", Num Survived: " << numSurvivors
    //          << ", P(Pclass | Survival): " << pPclassGivenSurvival << endl;

    //     // print non-survivor stats for the same pclass
    //     map<int, double>::iterator nonSurvivorIt =
    //         numNonSurvivorsGivenPclass.find(pclass);
    //     if (nonSurvivorIt != numNonSurvivorsGivenPclass.end()) {
    //         double numNonSurvivors = nonSurvivorIt->second;
    //         double totalNumObservationsForNonSurvivorsPclass =
    //             totalNumObservationsForPclass -
    //             totalNumObservationsForNonSurvivorsPclass;
    //         double pPclassGivenNonSurvival =
    //             numNonSurvivors / totalNumObservationsForNonSurvivorsPclass;
    //         cout << "Pclass: " << pclass
    //              << ", Num Non-Survivors: " << numNonSurvivors
    //              << ", P(Pclass | Non-Survival): " << pPclassGivenNonSurvival
    //              << endl;
    //     }
    // }

    // calculate p(sex|survival)

    map<double, double> numSurvivorsGivenSex;
    map<double, double> numNonSurvivorsGivenSex;
    for (int i = 0; i < sex_training.size(); i++) {
        double currSex = sex_training.at(i);
        if (numSurvivorsGivenSex.find(currSex) == numSurvivorsGivenSex.end()) {
            if (survived_training[i] == 1) {
                numSurvivorsGivenSex.insert(pair<double, double>(currSex, 1));
            } else {
                numNonSurvivorsGivenSex.insert(
                    pair<double, double>(currSex, 1));
            }
        } else {
            if (survived_training[i] == 1) {
                map<double, double>::iterator foundSexIterator =
                    numSurvivorsGivenSex.find(currSex);
                if (foundSexIterator != numSurvivorsGivenSex.end()) {
                    foundSexIterator->second += 1;
                }
            } else {
                map<double, double>::iterator foundSexIterator =
                    numNonSurvivorsGivenSex.find(currSex);
                if (foundSexIterator != numNonSurvivorsGivenSex.end()) {
                    foundSexIterator->second += 1;
                }
            }
        }
    }

    // for (map<double, double>::iterator it = numSurvivorsGivenSex.begin();
    //      it != numSurvivorsGivenSex.end(); ++it) {
    //     double sex = it->first;
    //     double numSurvivors = it->second;
    //     double numNonSurvivors = numNonSurvivorsGivenSex[sex];
    //     double totalNumObservationsForSex =
    //         count(sex_training.begin(), sex_training.end(), sex);
    //     double pSexGivenSurvival = numSurvivors / totalNumObservationsForSex;
    //     double pSexGivenNonSurvival =
    //         numNonSurvivors / totalNumObservationsForSex;
    //     cout << "Sex: " << sex << ", Num Survived: " << numSurvivors
    //          << ", Num Not Survived: " << numNonSurvivors
    //          << ", P(Sex | Survival): " << pSexGivenSurvival
    //          << ", P(Sex | Non-Survival): " << pSexGivenNonSurvival << endl;
    // }

    // calculate p(age)

    map<double, double> numObservationsForAge;
    for (int i = 0; i < age_training.size(); i++) {
        double currAge = age_training.at(i);
        if (numObservationsForAge.find(currAge) ==
            numObservationsForAge.end()) {  // the age has not been stored  yet

            // increment count for this age
            numObservationsForAge.insert(pair<double, double>(currAge, 1));

        } else {  // age has already been added
            // increment count for this age
            map<double, double>::iterator foundAgeIterator =
                numObservationsForAge.find(currAge);
            if (foundAgeIterator != numObservationsForAge.end()) {
                foundAgeIterator->second += 1;
            }
        }
    }

    // for (map<double, double>::iterator it = numObservationsForAge.begin();
    //      it != numObservationsForAge.end(); ++it) {
    //     double age = it->first;
    //     double numObservations = it->second;
    //     double totalNumObservations = age_training.size();
    //     double pAge = numObservations / totalNumObservations;
    //     cout << "Age: " << age << ", Num Observations: " << numObservations
    //          << ", P(Age): " << pAge << std::endl;
    // }

    // calculate p(pclass)

    map<int, double> numObservationsForPclass;
    for (int i = 0; i < pclass_training.size(); i++) {
        int currPclass = pclass_training.at(i);
        if (numObservationsForPclass.find(currPclass) ==
            numObservationsForPclass
                .end()) {  // the pclass has not been stored yet

            // increment count for this pclass
            numObservationsForPclass.insert(pair<int, double>(currPclass, 1));

        } else {  // pclass has already been added
            // increment count for this pclass
            map<int, double>::iterator foundPclassIterator =
                numObservationsForPclass.find(currPclass);
            if (foundPclassIterator != numObservationsForPclass.end()) {
                foundPclassIterator->second += 1;
            }
        }
    }

    // for (map<int, double>::iterator it = numObservationsForPclass.begin();
    //      it != numObservationsForPclass.end(); ++it) {
    //     int pclass = it->first;
    //     double numObservations = it->second;
    //     double totalNumObservations = pclass_training.size();
    //     double pPclass = numObservations / totalNumObservations;
    //     cout << "Pclass: " << pclass
    //          << ", Num Observations: " << numObservations
    //          << ", P(Pclass): " << pPclass << std::endl;
    // }

    // calculate p(sex)

    map<double, double> numObservationsForSex;
    for (int i = 0; i < sex_training.size(); i++) {
        double currSex = sex_training.at(i);
        if (numObservationsForSex.find(currSex) ==
            numObservationsForSex.end()) {  // the sex has not been stored yet

            // increment count for this sex
            numObservationsForSex.insert(pair<double, double>(currSex, 1));

        } else {  // sex has already been added
            // increment count for this sex
            map<double, double>::iterator foundSexIterator =
                numObservationsForSex.find(currSex);
            if (foundSexIterator != numObservationsForSex.end()) {
                foundSexIterator->second += 1;
            }
        }
    }

    // for (map<double, double>::iterator it = numObservationsForSex.begin();
    //      it != numObservationsForSex.end(); ++it) {
    //     double sex = it->first;
    //     double numObservations = it->second;
    //     double totalNumObservations = sex_training.size();
    //     double pSex = numObservations / totalNumObservations;
    //     cout << "Sex: " << sex << ", Num Observations: " << numObservations
    //          << ", P(Sex): " << pSex << std::endl;
    // }

    // end training
    chrono::high_resolution_clock::time_point trainingEnd =
        chrono::high_resolution_clock::now();

    chrono::high_resolution_clock::duration trainingTime =
        chrono::duration_cast<chrono::milliseconds>(trainingEnd -
                                                    trainingStart);
    cout << "Training time: " << trainingTime.count() << " milliseconds"
         << endl;

    int truePositives = 0;
    int trueNegatives = 0;
    int falsePositives = 0;
    int falseNegatives = 0;
    for (int i = numTrainingRows; i < numObservations; i++) {
        double age = age_testing.at(i - numTrainingRows);
        int pclass = pclass_testing.at(i - numTrainingRows);
        double sex = sex_testing.at(i - numTrainingRows);

        double pSurvivalGivenAge = numSurvivorsGivenAge[age] * priorSurvival;
        double pNonSurvivalGivenAge =
            numNonSurvivorsGivenAge[age] * priorNonSurvival;

        double pSurvivalGivenPclass =
            numSurvivorsGivenPclass[pclass] * priorSurvival;
        double pNonSurvivalGivenPclass =
            numNonSurvivorsGivenPclass[pclass] * priorNonSurvival;

        double pSurvivalGivenSex = numSurvivorsGivenSex[sex] * priorSurvival;
        double pNonSurvivalGivenSex =
            numNonSurvivorsGivenSex[sex] * priorNonSurvival;

        // calculate the posterior probability for survival and non-survival
        double pSurvival =
            pSurvivalGivenAge * pSurvivalGivenPclass * pSurvivalGivenSex;
        double pNonSurvival = pNonSurvivalGivenAge * pNonSurvivalGivenPclass *
                              pNonSurvivalGivenSex;

        // predict the class with maximum posterior probability
        int predictedClass = (pSurvival > pNonSurvival) ? 1 : 0;
        int trueClass = survived_testing[i - numTrainingRows];

        // update the confusion matrix
        if (predictedClass == 1 && trueClass == 1) {
            truePositives++;
        } else if (predictedClass == 0 && trueClass == 0) {
            trueNegatives++;
        } else if (predictedClass == 1 && trueClass == 0) {
            falsePositives++;
        } else {
            falseNegatives++;
        }
    }
    int numTestingRows = id_vector.size() - numTrainingRows;

    double accuracy = (truePositives + trueNegatives) / (double)numTestingRows;
    double precision = truePositives / (double)(truePositives + falsePositives);
    double recall = truePositives / (double)(truePositives + falseNegatives);
    double f1Score = 2 * precision * recall / (precision + recall);

    cout << "Accuracy: " << accuracy << endl;
    cout << "Precision: " << precision << endl;
    cout << "Recall: " << recall << endl;
    cout << "F1 Score: " << f1Score << endl;

    return 0;
}
