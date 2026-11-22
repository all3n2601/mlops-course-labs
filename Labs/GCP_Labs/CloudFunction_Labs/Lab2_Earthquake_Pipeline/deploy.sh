#!/usr/bin/env bash
# The commands I ran, in order. Set PROJECT, ACCOUNT and BUCKET first. Nothing here holds a secret.
set -euo pipefail
PROJECT=${PROJECT:?}; ACCOUNT=${ACCOUNT:?}; PROJECT_NUMBER=${PROJECT_NUMBER:?}
REGION=us-central1
BUCKET=lab6-eq-pipeline-$PROJECT_NUMBER
SA=lab6-pipeline@$PROJECT.iam.gserviceaccount.com
F=(--project="$PROJECT" --account="$ACCOUNT")

gcloud services enable cloudfunctions.googleapis.com cloudbuild.googleapis.com run.googleapis.com \
  eventarc.googleapis.com pubsub.googleapis.com workflows.googleapis.com workflowexecutions.googleapis.com "${F[@]}"

gcloud storage buckets create gs://$BUCKET --location=$REGION --uniform-bucket-level-access "${F[@]}"
gcloud iam service-accounts create lab6-pipeline --display-name="lab6 pipeline" "${F[@]}"
gcloud pubsub topics create lab6-model-training-trigger "${F[@]}"

gcloud storage buckets add-iam-policy-binding gs://$BUCKET --member=serviceAccount:$SA --role=roles/storage.objectAdmin "${F[@]}"
gcloud pubsub topics add-iam-policy-binding lab6-model-training-trigger --member=serviceAccount:$SA --role=roles/pubsub.publisher "${F[@]}"
# project-level roles (remove them again when you clean up)
for role in roles/eventarc.eventReceiver roles/serviceusage.serviceUsageConsumer; do
  gcloud projects add-iam-policy-binding "$PROJECT" --member=serviceAccount:$SA --role=$role --condition=None --account="$ACCOUNT"
done
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member=serviceAccount:service-$PROJECT_NUMBER@gs-project-accounts.iam.gserviceaccount.com \
  --role=roles/pubsub.publisher --condition=None --account="$ACCOUNT"

gcloud functions deploy lab6-train-model --gen2 --runtime=python312 --region=$REGION --source=src/train_model \
  --entry-point=train_model --trigger-topic=lab6-model-training-trigger --trigger-service-account=$SA \
  --service-account=$SA --memory=1GiB --timeout=540s --set-env-vars=BUCKET=$BUCKET "${F[@]}"
gcloud functions deploy lab6-process-data --gen2 --runtime=python312 --region=$REGION --source=src/process_data \
  --entry-point=process_data --trigger-bucket=$BUCKET --trigger-service-account=$SA --service-account=$SA \
  --memory=1GiB --timeout=300s --set-env-vars=GCP_PROJECT=$PROJECT,TRAIN_TOPIC=lab6-model-training-trigger "${F[@]}"
gcloud functions deploy lab6-predict --gen2 --runtime=python312 --region=$REGION --source=src/predict \
  --entry-point=predict --trigger-http --no-allow-unauthenticated --service-account=$SA \
  --memory=512MiB --timeout=60s --set-env-vars=BUCKET=$BUCKET "${F[@]}"

# a custom trigger service account is not granted run.invoker automatically
for fn in lab6-process-data lab6-train-model lab6-predict; do
  gcloud run services add-iam-policy-binding $fn --region=$REGION --member=serviceAccount:$SA --role=roles/run.invoker "${F[@]}"
done

gcloud workflows deploy lab6-earthquake-pipeline --source=workflow.yaml --location=$REGION --service-account=$SA "${F[@]}"

# run it: upload the data (fires the trigger once by itself), then run the workflow
gcloud storage cp data/usgs_earthquakes_2025.csv gs://$BUCKET/raw/usgs_earthquakes_2025.csv "${F[@]}"
PREDICT_URL=$(gcloud functions describe lab6-predict --gen2 --region=$REGION --format="value(serviceConfig.uri)" "${F[@]}")
gcloud workflows run lab6-earthquake-pipeline --location=$REGION "${F[@]}" \
  --data="{\"bucket\":\"$BUCKET\",\"raw_object\":\"raw/usgs_earthquakes_2025.csv\",\"predict_url\":\"$PREDICT_URL\"}"
