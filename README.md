# Text Classification Pipeline with Deep Learning Models

## Overview

This project implements a pipeline for text classification using state-of-the-art Deep Learning models. The pipeline facilitates importing data from CSV files containing text samples along with their corresponding labels. It leverages pre-trained models available in the Hugging Face Hub and provides functionality to retrain these models using custom datasets. Additionally, the pipeline integrates explainable AI algorithms based on the Captum library, enabling users to understand the impact of individual words on model predictions.

## Features

- **Data Import**: Import text data and labels from CSV files.
- **Model Integration**: Easily import pre-trained models from the Hugging Face Hub.
- **Retraining**: Retrain imported models using custom datasets for domain-specific tasks.
- **Explainable AI**: Utilize Captum library algorithms to interpret model decisions.
- **Layer Integrated Gradients**: Employ [Layer Integrated Gradients](https://medium.com/@kevinkhang2909/xai-use-captum-to-deep-dive-sentiment-analysis-86b46bff092b) to understand word-level contributions to predictions.
- **MLflow Integration**: Run and track experiments with MLflow, including an example in the `examples/mlflow` directory for training Large Language Models (LLMs) for text classification.

## Getting Started

To get started with the text classification pipeline, follow these steps:

1. **Installation**: Clone the repository to your local machine.

    ```
    git clone https://github.com/GutoL/Deep-Learning-Text-Pipeline.git
    ```

2. **Dependencies**: Install the required dependencies using pip.

    ```
    pip install -r requirements.txt
    ```

3. **Usage**: Refer to the provided examples to understand how to use the pipeline effectively.

## Examples

Explore the `examples` directory for sample scripts demonstrating various use cases of the text classification pipeline. This includes examples for using MLflow for tracking and training LLMs for text classification in `examples/mlflow`.

## Graphic interface

This pipeline can be performed by a graphic interface tool, that was developed using [streamlit](https://streamlit.io/). To run this app follow the next steps:

```bash
cd streamlit
streamlit run Homepage.py
```
A web page will open:


https://github.com/user-attachments/assets/4bde73e2-6338-4614-8980-ef4536576258

## Technologies Used
The project makes use of the following technologies:

- **Python**: Core language for developing the pipeline.
- **Hugging Face Transformers**: For importing and using pre-trained models.
- **Captum**: For implementing explainable AI methods.
- **MLflow**: For experiment tracking and model lifecycle management.
- **Streamlit**: For building the graphical user interface.
- **Pandas**: For data manipulation.
- **PyTorch**: As the deep learning framework for model training.

## Contributing

Contributions to the project are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the Apache License 2.0 License. See the `LICENSE` file for more details.

## Acknowledgments

- We thank the [Hugging Face Hub](https://huggingface.co/docs/hub/index) community for providing pre-trained models and valuable resources.
- Special thanks to the [Captum](https://github.com/pytorch/captum) development team for their explainable AI library.
