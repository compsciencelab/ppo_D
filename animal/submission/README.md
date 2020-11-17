# Submission example

We provide here the code for creating a submission container, as well as a very basic agent trained on `1-Food` using `trainMLAgents.py`. The `agent.py` implements the 
script you will need to submit, it loads a trained model located in `data`.
Read the [submission documentation](../../documentation/submission.md)

## Preliminaries to do once

```
  pip install evalai
  evalai set_token 5efae622e235df8963044a105b4d4a6d797fd02c
```

## Local tests
This is to test as close as possible as the real test environment but on our testset. 
```
    docker run -v "$PWD"/test_submission:/aaio/test -v "$PWD":/myaaio submission python /aaio/test/testDockerLab.py 
```

## Submit
Everything is in animal/submission/. Copy the new network which must be named ''animal.state_dict'' into the animal/submission/data/ folder.

Create new image for submission
```
  docker image rm -f submission
  ./update.sh
  docker build --tag=submission .
```

Test it
```
  docker run -v "$PWD"/test_submission:/aaio/test submission python /aaio/test/testDocker.py
  #docker run --gpus all -v "$PWD"/test_submission:/aaio/test submission python /aaio/test/eval_train_docker.py
```

If the results says success, then it's ok and go on to submit.

Submit it

```
  evalai push submission:latest --phase animalai-main-396
```

## Updates
```
   docker run -it -v "$PWD"/test_submission:/aaio/test -v "$PWD":/aaio/base submission /bin/bash
   cp base/agent.py .
   cp base/data/animal.state_dict data/
```
From other shell:
```
   docker ps # to see container id
   docker commit <container id> submission
```

