# uva-mlprod-code
## Week 4 - Deployment


### How to Run Things

#### 1. Kill all docker containers.

Starting fresh

Stop any containers started by docker compose
```shell
docker compose -f infra/docker-compose.yaml down
```

Rebuild the `nova` image
```shell
make build
```

#### 2. Start MLFlow and Airflow

The application will look at the Alpha experiment and will try to fetch the model from its artifacts. As MLFlow is empty now, it will throw an error if we try to run the app. 

So what we will do is to bring up mlflow so we can start an experiment there. Also, we will bring up the Airflow, so that we can run the training DAG there, which will train and push the model to MLFlow.

```shell
docker compose -f infra/docker-compose.yaml up mlflow airflow
```

In order to free the terminal, we can hit `d` to Detach the services. It should be actually visible as a tip at the bottom of the terminal.

Mind that it takes a while for Airflow to go up. Check `docker ps` and wait till both services are in `healthy` status.

#### 3. Start the Training DAG

Now time to train a model and push to MLFlow. For that, we already have a DAG. Moreover, that DAG file should already been mounted to Airflow's storage. Therefor we only need to do a few clicks to start the DAG.

* Go to http://localhost:4242/.
* Go to `Dags` on the left pane.
* Go to `nova_training`.
* Trigger the DAG with the `Trigger` button on the top.
* Select the Single run and make sure `Unpause` is on.
* Trigger and wait for all the tasks to complete.

If everything is green, the model should be in mlflow.


#### 4. Check if the model is in MLFlow

After the DAG completes successfully, we should be able to see the model in MLFlow.

* Go to `http://localhost:8080/#/experiments`
* Switch to the "Model training" view
* Go to Experiment `Alpha`
* Go to the latest run.
* Go to the Artifacts.

Are all the files there? Cool!

#### 5. Start the application

Now that we have all the dependencies ready, let's start the server.

```shell
docker compose -f infra/docker-compose.yaml up app
```

The server should start and like other dag's it will block the terminal. Again you can detach it but maybe don't so you can see the logs.

* Go to http://localhost:5001/health and see OK text.
* Run ` sh scripts/api_request_example.sh` which will send a request to the server and print the result.
* See the result `false`. Check the server logs and see `"POST /predict HTTP/1.1" 200`. 

All good? You have just deployed an E2E ML backed API!
