#!/bin/sh

# Regressió logística
curl --request POST "http://localhost:8000/classify" --header "Content-Type: application/json" --data-raw "{\"model_type\": \"lr\", \"data\": [ 1, 1 ]}"
curl --request POST "http://localhost:8000/classify" --header "Content-Type: application/json" --data-raw "{\"model_type\": \"lr\", \"data\": [ 5, 4 ]}"

# Màquina de suport vectorial
curl --request POST "http://localhost:8000/classify" --header "Content-Type: application/json" --data-raw "{\"model_type\": \"svm\", \"data\": [ 1, 1 ]}"
curl --request POST "http://localhost:8000/classify" --header "Content-Type: application/json" --data-raw "{\"model_type\": \"svm\", \"data\": [ 5, 4 ]}"

# Arbres de decisió
curl --request POST "http://localhost:8000/classify" --header "Content-Type: application/json" --data-raw "{\"model_type\": \"tree\", \"data\": [ 1, 1 ]}"
curl --request POST "http://localhost:8000/classify" --header "Content-Type: application/json" --data-raw "{\"model_type\": \"tree\", \"data\": [ 5, 4 ]}"

# K veïnats més propers
curl --request POST "http://localhost:8000/classify" --header "Content-Type: application/json" --data-raw "{\"model_type\": \"knn\", \"data\": [ 1, 1 ]}"
curl --request POST "http://localhost:8000/classify" --header "Content-Type: application/json" --data-raw "{\"model_type\": \"knn\", \"data\": [ 5, 4 ]}"
