# Used Car Price Prediction Machine Learning Project

This repository contains the implementation of a **Used Car Price Prediction** machine learning model using data scraped from **Cars24**, a leading platform for buying and selling used cars. The project spans the entire Machine Learning lifecycle, including data scraping, data cleaning, feature engineering, model building, evaluation, and deployment. Additionally, it incorporates **continuous model training and retraining** using **Airflow** for automation, as well as MLOps tools for version control, model management, and pipeline orchestration.

## Project Overview

The objective of this project is to build a machine learning model capable of predicting the price of a used car based on various features such as brand, model, year of manufacture, mileage, and more. Data was scraped from **Cars24**, cleaned, preprocessed, and used to train a machine learning model for price prediction. The project also implements continuous training using **Apache Airflow** to periodically retrain the model with new data.

## Key Components

1. **Data Scraping**:
   - Data was scraped from **Cars24** using **web scraping techniques** (using libraries like `requests` and `BeautifulSoup`).
   - The dataset contains information such as car make, model, year, mileage, transmission type, fuel type, and car condition.

2. **Data Cleaning & Preprocessing**:
   - Missing values were handled, irrelevant columns were removed, and outliers were identified and treated.
   - Data types were corrected, categorical variables were encoded, and feature scaling was applied where necessary.

3. **Exploratory Data Analysis (EDA)**:
   - Performed thorough exploratory data analysis (EDA) to understand the distribution of car prices and identify patterns or correlations between features and price.
   - Visualizations like histograms, boxplots, and scatter plots were used to explore relationships.

4. **Feature Engineering**:
   - Created new features based on existing data (e.g., extracting car age from the year of manufacture, converting textual features into numerical values like car condition).
   - Used one-hot encoding for categorical features (e.g., fuel type, transmission type).

5. **Model Building**:
   - Built machine learning models using **Random Forest**, **Gradient Boosting**, and **XGBoost** to predict car prices.
   - Hyperparameter tuning and cross-validation were performed to optimize model performance.

6. **Model Evaluation**:
   - Evaluated models using performance metrics like **RMSE**, **MAE**, and **R-squared**.
   - Compared different models to determine the best performing model for price prediction.

7. **Continuous Training with Apache Airflow**:
   - Implemented continuous model training using **Apache Airflow** to automate the process of data scraping, model training, and model evaluation.
   - Airflow DAGs (Directed Acyclic Graphs) were used to schedule and monitor the training pipeline, ensuring the model is retrained periodically with fresh data.

8. **Model Deployment with MLOps**:
   - Deployed the trained model using **MLFlow** for model tracking, versioning, and management.
   - Containerized the application with **Docker** for easy deployment and scalability.
   - Used **Flask** to serve the model as an API, allowing users to input car details and get price predictions.

9. **Data Version Control (DVC)**:
   - Utilized **DVC** for version control of the datasets and models, enabling reproducibility and tracking of the model’s evolution.

## Technologies Used

- **Data Collection**: Python, Requests, BeautifulSoup
- **Data Processing**: Pandas, Numpy, Scikit-learn
- **Machine Learning Models**: Scikit-learn, XGBoost, LightGBM
- **Model Evaluation**: Matplotlib, Seaborn
- **MLOps Tools**: Git, DVC, MLFlow, Docker, Airflow
- **Deployment**: Flask, Docker
- **Pipeline Orchestration**: Apache Airflow

## How to Run the Project

### Prerequisites:
- Python 3.x
- Docker
- Git
- DVC
- MLFlow
- Apache Airflow

### Installation:
1. Clone the repository:
   ```bash
   git clone https://github.com/username/used-car-price-prediction.git
   cd used-car-price-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up DVC and pull data:
   ```bash
   dvc pull
   ```

4. Train and evaluate the model:
   ```bash
   python train_model.py
   ```

5. Run the API to make predictions:
   ```bash
   python app.py
   ```

### Docker Setup:
You can also run the entire project in a Docker container:
```bash
docker build -t used-car-price-prediction .
docker run -p 5000:5000 used-car-price-prediction
```

### Running Continuous Training with Airflow:
To set up the automated training pipeline with Airflow, follow these steps:
1. Start the Airflow web server and scheduler:
   ```bash
   airflow webserver -p 8080
   airflow scheduler
   ```

2. Trigger the Airflow DAG for continuous training:
   - Access the Airflow UI at `http://localhost:8080` to monitor the pipeline.

3. The DAG will automate the data scraping, training, and evaluation of the model periodically.

## Future Enhancements

- Integrate external data sources for better prediction accuracy, such as car maintenance history or external market trends.
- Improve model interpretability with techniques like SHAP or LIME to understand the impact of individual features on predictions.
- Implement a model monitoring system to track performance over time and retrain when necessary.
- Expand the deployment pipeline to support cloud platforms such as AWS, GCP, or Azure for scalable deployment.

## Contributing

Feel free to fork the repository and submit pull requests. Contributions to improving the model, adding new features, or enhancing the MLOps pipeline are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



========================================================= Old Description ======================================
# end-to-end-used-car-price-prediction

![alt text](image.png)

![alt text](image-1.png)

**MLOps == ML + Ops**

**ML**
1. Data Gathering
2. EDA
3. FE (data transformation)
4. Model building
5. Model evaluation

**Ops**
6. Deploy
7. Maintain & Monitor
8. Retraining



`Step-By-Step-MLOps-Project-Workflow`

1. Create a GitHub repository for the project using the "Python" .gitignore template and MIT License. Then, clone the repository to local system: git clone <repo URL>

2. Create project file/folder structure using template.py OR we can create this project repo by cloning from template repo
    git repo url: `https://github.com/iamprashantjain/end-to-end-used-car-price-prediction`

![alt text](image-2.png)

3. Continuous Integration (CI) is a software development practice where developers frequently integrate their
code into a shared repository multiple times a day. Each integration (or "commit") is automatically tested to detect
issues early, enabling teams to identify and address bugs as soon as they arise. The goal of CI is to improve software
quality and reduce the time it takes to deliver updates.

4. - .github folder contains workflows where we will keep config files for CI
   - experiment is a jupyter file where we will perform eda, transformations etc, we can create seperate files also
   - src is the main folder where we will keep all components of machine learning like training, transformation, model building & evaluation and pipelines like prediction & training pipeline. we will also keep exception, logger & utils files in this folder
   - test folder is for testing the code, single unit test & integration testing
   - .gitignore is the file which has information of all files which should be ignored while pushing code files to github
   - init_setup.sh is the sheel script where we can write all cmd commands like venv activation etc
   - pyproject.toml, setup.py & setup.cfg are all configuration files
   - tox.ini is the fils to perform testing automatically as soon as we push the code to github
   - 2 requirement files: for development & for production


5. There is no need to upload code to pypi unless you want others to use your repo in thier code, ml projects are of no use for others so we dont need to upload that code to pypi

6. Docker to containerize the code

![alt text](image-3.png)

![alt text](image-4.png)

![alt text](image-5.png)

![alt text](image-6.png)

![alt text](image-7.png)

7. `No we can begin with the project implimentation and we'll learn other tools in between like MLFlow, DVC, Airflow`

8. We can also incorporate webscraping the data `ETL Pipeline` and use that data in data ingestion, transformation, model building, evaluation and then use MLOps tools to deploy, maintain and monitor

9. Currently, We will impliment end to end machine learning project:
    1. Building the project `git, github, python`
    2. Test: unit & integrated `pytest, tox`
    3. Deliver the project using `docker image on github action server saved on azure`
    3. Deploy using (CI/CD concept) `All above (CI) will be done on Github Action Server`
    4. Monitor the project `Evidently.AI`
    5. Retraining `(CT concept) using Airflow`
    6. `DVC` for data management
    7. `MLFlow` experiment tracking 
    8. `Dagshub/BentoML` for model registry

10. We will start with Jupyter notebook implimentation of the project
11. Perform modular coding
12. Adding variuos tools
13. Deployment


**Project Description**
- This project involves building an end-to-end machine learning pipeline for predicting used car prices. The process will include:

1. Web Scraping: Automating the extraction of used car data from the Cars24 website.
2. Model Development: Building and training a machine learning model to predict car prices based on the scraped data.
3. Deployment: Deploying the model in a production environment using Docker and GitHub Actions.
4. Monitoring & Retraining: Continuously monitoring model performance and automatically retraining the model as new data is scraped.

- The goal is to streamline the entire workflow, from data collection to model deployment and ongoing improvements.

14. Update setup.py file & Update `init_setup.sh` script to execute all commands in 1 shot, use `bash init_setup.sh` to run the script in bash terminal (if venv not activated then activate it manually since source activate is for gitbash/linux system not windows powershell or cmd)

15. Initiate the project implementation in a Jupyter notebook (experiments.ipynb), or alternatively, organize and document the project files. Ensure that you outline and define all key steps, including data cleaning, transformation, feature engineering, and model selection, as part of the experimentation process. This will make it easy to convert into modular coding and also at the time of re-training model with new data.


16. Once everything is experimented in jupyter, we can start write modular code

17. Write code for exception and logging

18. components -- ![alt text](image-8.png)

In each component: DI, DT, MT, ME there are 2 things:
    1. config: path of the output
    2. artifcats: output

- for example: In data ingestion config, we will define path of the output files and then whatever will be the output after the data ingestion process, it will be saved on those paths and will be used by next component. In data transformation, it will read output data from the paths defined in config of data ingestion and perform transformation and save the preprocessor in the path defined in data transformation config and so on.. 

19. Write code for data ingestion and incorporate intial data cleaning & feature engineering here (except sclaing, encoding & programatic missing value imputation), also test run the script to make sure everything is working fine.

20. Next, Write modular code for data transformation which creates preprocessor.pkl file to, again test run the script to make sure everything is working fine.

21. Since Data have both categorical & nuerical data with missing value in categorical data. There are 3 options to impute missing values:
    1. Subject Matter Expertise & requires manually filling missing values, which is not possible bcoz I want to automate everything and also missing values are huge so it will be imposible

    2. Impute with "most frequent" values: It will create bias for columns where some categories are more means in the case of data imbalance

    3. KNN Imputation: This requires values to be encoded since it doesnt work on strings, so encoding rows which has missing values might give incorrect results but thats the better option right now since it can handle complex relationship

    4. So using KNN Imputation for both Numerical & Categorical data

22. Continously facing issue in transformation key error "content.onRoadprice" :: check logs for that, applied below solution:

    - The original code was passing X['content.onRoadPrice'] (the target column) directly as an argument to select_best_k. However, select_best_k expects y (the target variable), and X (the feature set) separately. This means X[num_cols] should be passed as the feature set, and the target column (e.g., y_train) should be passed to the method separately. In the updated code, I corrected this by passing y_train (the target values) from the initialize_data_transformation method into select_best_k instead of referencing the target column directly from X.


23. We have preprocessor.pkl saved in artifacts, now we can write code for "model_training"

24. Write code for model_trainer & then write code for training_pipeline incorporating input & output from all components. (getting train data & test data path error, while debugging i noticed i was returning dictionary with train data & test data path so fixed it to only return the path & not dictionary, which started working)

25. After model training, best model found which we are going to productionized. We can now write code for prediction_pipeline and to get the input data, write code for simple flask application.

26. While Displaying predictions, I've noticed that instead of showing absolute amount, We should disply the range of price based on 95% CI. So I've written a function inside utils.py file which is used in prediction pipeline to predict the 95% CI range (also updated model_trainer.py file to return the algorithm selected for model building to use while calculating 95% CI)

27. Run the training_pipeline again & then to test the prediction pipeline, Run flask app

28. Now, Its time to add MLOps tools to the project like MLFlow, DVC, DAGS, Airflow, Docker etc

29. 1st Tool: MLFlow for experiment tracking. Append this under src/components - add a seperate file `model_evaluation.py` for experiment tracking or we can also add mlflow tracking mechanism under `model_trainer.py`and after that check it by executing command `mlflow ui`

![alt text](image-9.png)

30. 2nd Tool: DVC (data version control) which we can use to manage different versions of data

![alt text](image-11.png)

![alt text](image-12.png)

![alt text](image-13.png)

31. We can configure pipeline reproducibility using dvc.yaml file as shown above.

32. 3rd Tool: `Docker: To create Containers`

33. 4th Tool: `Airflow: To automate continous training & Schedule ETL pipelines`

34. **Current Architecture**

![alt text](image-15.png)

    -   For our machine learning project, We need to do:
        
        1. Training
            1. Perform DI, DT, MT & Save All artifcats objects (preprocessor & model) to repo
        2. Prediction
            1. Batch prediction to take bulk data (create a dag inside the same airflow container)
            2. Single Value (We have created Flask app which we can run in a different container)

    - `For Continous training, We are using Airflow to schdeule on h/w/m/y basis`, which will automatically perform training & save object in artifacts folder. To use airflow, we have created a docker image inside the container to run airflow which is listening to port 8080.

35. Create 2 files: Dockerfile.airflow (copy of Dockerfile previosuly created) & Dockerfile.flask (To run flask app), Use `docker-compose build --no-cache && docker-compose up` to build & run the airflow 


36. Setup cloud to save artifacats & in batch_prediction

37. Deploy docker image to Cloud: AWS/Azure/GCP

==========================================================================

Pre-Requisite: Python, Docker

1. To Test Run Project, Simply clone this repository: `git clone https://github.com/iamprashantjain/end-to-end-machine-learning-project.git`
2. Activate virtual environment: `.\venv\Scripts\activate`
3. Once activated, run: `docker-compose build --no-cache && docker-compose up`. This will build & up the docker container with 2 images:
    - Airflow: access it from localhost:8080
    - Flask app: access it from localhost:80/predict
4. Cleanup everything in Docker:
    - `docker system prune -a -f --volumes`
