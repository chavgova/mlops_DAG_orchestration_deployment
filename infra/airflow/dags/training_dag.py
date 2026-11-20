from airflow import DAG
from docker.types import Mount
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator

from datetime import timedelta
import pendulum


default_args = {
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

# automatic run every day at 23:59
dag = DAG(
    dag_id="nova_training",
    default_args=default_args,
    schedule="59 23 * * *",
    start_date=pendulum.datetime(2025, 1, 1, tz="Europe/Amsterdam"),
    catchup=False,
    is_paused_upon_creation=False,
)



preprocessing_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/main_training.py --preprocess",
    image="nova:latest",
    network_mode="infra_default",
    task_id="preprocessing",
    mount_tmp_dir=False,
    mounts=[
        Mount(
            source="data", target="/nova/data"
        )  # note: Normally we do not do this via local files. I would use a cloud storage system like AWS S3 or Azure Blob Storage.
    ],
    dag=dag,
)

feat_eng_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/main_training.py --feat-eng",
    image="nova:latest",
    network_mode="infra_default",  # note: In order services to communicate each other, we add them to the same network. You do not see in docker-compose.yaml, b/c by default it puts all the services under one network called "infra_default".
    task_id="feature_engineering",
    mount_tmp_dir=False,
    mounts=[Mount(source="data", target="/nova/data")],
    dag=dag,
)


model_training_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/main_training.py --training",
    image="nova:latest",
    network_mode="infra_default",
    task_id="training",
    mount_tmp_dir=False,
    mounts=[Mount(source="data", target="/nova/data")],
    dag=dag,
)


(
    preprocessing_task >>
    feat_eng_task >> 
    model_training_task
)
