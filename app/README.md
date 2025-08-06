## FASTAPI app to run app
1. pip install -r requirements.txt
2. Deploy to cloud run
```
gcloud run deploy
--source .
--port=8080
--service-account=SERVICE_ACCOUNT
--no-cpu-throttling
--min-instances=0
--region=us-central1
--project=PROJECT
```


## Test the container locally 
1. docker build -t fastapiserver .  
2. docker run -it --rm fastapiserver