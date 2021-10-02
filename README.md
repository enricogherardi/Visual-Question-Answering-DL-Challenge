# Visual Question Answering DL Challenge

In this repository you can find all the material used to take part at the competitions created for the Artifical Neural Networks and Deep Learning exam at Politecnico di Milano.
The goal of this challenge is to build the best model to solve a Visual Question Answering.
The dataset is composed by synthetic scenes, in which people and objects interact, and by corresponding questions, which are about the content of the images. Given an image and a question, the goal is to provide the correct answer. Answers belong to 3 possible categories: 'yes/no', 'counting' (from 0 to 5) and 'other' (e.g. colors, location, ecc.) answers.

## Our Best Model

The final model multiplies a CNN (MobileNetV2) with a Transformer-based net and connects it to a FFNN to classify the right answer to corresponding the couple (image, question).

## Conclusion

For a more in-depth analysis I recommend you to read the report of the challenge. The final evaluation of this project by our professor was 2.0/2.0.