from locust import HttpUser, task, between, events
from locust.env import Environment
import os
import random


test_data_folder = '/Users/mschulze/Downloads/load_test_data'

characters = {'Leo': ['coxet', '1'], 
            'Lila': ['wythoff', '2'], 
            'Ethan': ['hanner', '3'], 
            'Axel': ['hedron', '4']}


@events.init.add_listener
def on_locust_init(environment: Environment, **kwargs: int) -> None:
    environment.filenames = os.listdir(images_folder)


class QuickstartUser(HttpUser):
    wait_time = between(1,5)  #user waits rand 1-5 secs between requests

    @task
    def call_root_endpoint(self) -> None:
        self.client.get('/')
    
    # 3 is the weight for rand prob. So since 1 task hits root endpoint and 3 hit predict... 
    # 1/4 prob test root and 3/4 prob test predict
    @task(3) 
    def call_predict(self) -> None:
        self.client.post(
                '/predict',
                data={},
                files=[("", , ""))],
        )

    
    def get_random_char(self) -> str:
        return random.choice(characters.keys())
    
    def get_random_question(self) -> str:
        return random.choice()


# run using this... locust -f locust_test.py
