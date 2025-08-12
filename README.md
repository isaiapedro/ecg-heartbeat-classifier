# ECG Heartbeat Classifier
### (Ensemble Neural Network - ANN, CNN & LSTM)

## Contents

- [How to run](#how-to-run)
- [Introduction](#introduction)
- [Data Overview](#data-overview)
- [Results](#results)
- [Conclusion](#conclusion)
  
## How to Run

<br>

With Docker installed, run Docker Desktop and the jupyter notebook image with the following command

<br>

```
docker run -it --rm -p 8888:8888 -v "${pwd}:/tf/notebooks" tensorflow/tensorflow:latest-jupyter
```
## Introduction

This repository was made to build a classifier that finds medical patterns based of eletrocardiografic signals with great precision, using the [MIT-BIH Arrhythmia Dataset and The PTB Diagnostic ECG Database](https://www.kaggle.com/datasets/shayanfazeli/heartbeat).

<br/>

Later on, this model will be used to predict streamed medical data with the help of a database and the frameworks Spark Streaming and Kafka. The trustworthiness of the model will be tested using different datasets and improved according to state of the art practices.

<br/>


## Data Overview



## Results



## Conclusion

Thanks for reading up until here. I had a ton of fun doing this notebook and got a lot of useful insights on Convolution and how to setup a Neural Network with three hidden layers that acts like filters.

If you want to see more Kaggle solutions, see the Flower Classification Problem or go to my github page. Feel free to reach me on [LinkedIn](https://www.linkedin.com/in/isaiapedro/) or my [Webpage](https://github.com/isaiapedro/Portfolio-Website).

Bye! 👋
