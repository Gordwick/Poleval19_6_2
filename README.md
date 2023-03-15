author: Sebastian Sitko

# HerBerta Classification model for task 6.2 from PolEval 2019 with deployment service in FastAPI

### Task: Type of harmfulness

In this task, the participants shall distinguish between three classes of tweets: 0 (non-harmful), 1 (cyberbullying), 2 (hate-speech). There are various definitions of both cyberbullying and hate-speech, some of them even putting those two phenomena in the same group. 


### Build and run docker:

* docker build --compress -r ml/message_classification_service .

* docker run -p 8080:80 -e "PYTHON_ENV=development" --name="message_classification_run" ml/message_classification_service:latest


### Testing endpoint:
* pytest ./tests
