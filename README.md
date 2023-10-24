# Deep Learning Text Pipeline

This repository contains the implementation of a pipeline for text classification using Deep Learning models. The pipeline imports data as a CSV file with text and the respective labels. 
The pipeline is able to import pre-trained models from the [Hugging Face Hub](https://huggingface.co/docs/hub/index) and retrain them using the specified dataset. 
The pipeline also is able to run explainable AI algorithms based on the [Captum library](https://github.com/pytorch/captum). Currently, we use the algorithm [Layer Integrated Gradients](https://medium.com/@kevinkhang2909/xai-use-captum-to-deep-dive-sentiment-analysis-86b46bff092b) in order to understand the impact of each word into the prediction made by the model.
