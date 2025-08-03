## Deplying a Gemma3n model for inference
1. Deploy the model
```
gcloud run deploy gemmatest \
   --image us-docker.pkg.dev/cloudrun/container/gemma/gemma3-4b \
   --concurrency 4 \
   --cpu 8 \
   --set-env-vars OLLAMA_NUM_PARALLEL=4 \
   --max-instances 1 \
   --memory 32Gi \
   --allow-unauthenticated \
   --no-cpu-throttling \
   --timeout=600 \
   --region us-central1
```
2. Test the model:
    a. Cloud Run url looks like: name-111111.region.run.app
```
curl {CLOUD_RUN_URL}/api/generate -d '{
  "model": "gemma3:4b",
  "prompt": "Why is the sky blue?"
}'
```

https://cloud.google.com/run/docs/run-gemma-on-cloud-run
https://cloud.google.com/run/docs/quickstarts/jobs/build-create-python