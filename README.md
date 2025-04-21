# GMM-option-orders
A pipeline based on Gaussian mixture models to process various outputs from large language models caused by different option orders, which can perform classification, multiple-choice question and abstention tasks

The file named `GLM4.py`, `llama2.py` and `llama3.py` are related to utilize the corresponding LLM to perform the task of classification or multiple-choice question, where the processes are similar to each other. Specifically, the detailed annotations in English are provided in `classification scenario/GLM4.py`.

The file named `GMM_pipeline.py` implements our GMM-based pipeline, including the coefficients estimation and the performance comparison with other existing methods. Furthermore, in the folder `MCQ scenario`, the coefficients in our pipeline that used for all 4-option MCQ datasets is estimated based on the `GMM_pipeline(MMLU).py`, which further conducts the evaluation on MMLU; the evaluation on other 4-option MCQ datasets are conducted in `GMM_pipeline(4-options).py`. For the only 5-option MCQ dataset CSQA, the estimation and evaluation are performed in `GMM_pipeline(CSQA).py`. 

The performance comparison on abstention task is performed in `MCQ scenario/Abstention pipeline.py`.

The original dataset and the corresponding evaluation results are provided in `Data`. 
