#!/usr/bin/env bash
# One-time Azure setup for the CI/CD pipeline in .github/workflows/ci-cd.yml.
#
# Creates one resource group containing:
#   - Azure Container Registry: stores the Docker images, one tag per git commit
#   - Container Apps environment and app: serves the dashboard over HTTPS and scales to zero when idle
#   - two managed identities: one the app pulls images with, one GitHub Actions signs in as
#     (OIDC federation, so no Azure password is stored in GitHub)
# and saves the IDs the pipeline needs as GitHub repository secrets and variables.
#
# Needs az (signed in with `az login`), docker and gh (signed in). Run from the repo root:
#     bash deploy/azure_setup.sh
# Delete everything afterwards with:
#     az group delete --name rg-ctr-dashboard --yes
set -euo pipefail

# Azure for Students allows only a few regions (list them with `az policy assignment list`) and one
# Container Apps environment per subscription. East Asia provisions in about two minutes.
LOCATION="${LOCATION:-eastasia}"
RG="${RG:-rg-ctr-dashboard}"
APP="${APP:-ctr-dashboard}"
ENVIRONMENT="${ENVIRONMENT:-ctr-dashboard-env}"
IMAGE_NAME="ctr-dashboard"
# The first image is built from this commit (what is on GitHub), never from uncommitted local edits.
SOURCE_REF="${SOURCE_REF:-origin/main}"

REPO="$(gh repo view --json nameWithOwner --jq .nameWithOwner)"
# Subject prefix of GitHub's OIDC tokens for this repo. Newer repos include owner and repo IDs
# (repo:owner@123/name@456); older ones use repo:owner/name.
SUBJECT_PREFIX="$(gh api "repos/$REPO/actions/oidc/customization/sub" --jq '.sub_claim_prefix // empty' || true)"
SUBJECT_PREFIX="${SUBJECT_PREFIX:-repo:$REPO}"
SUBSCRIPTION_ID="$(az account show --query id --output tsv)"
TENANT_ID="$(az account show --query tenantId --output tsv)"
# Registry names are global and must be 5-50 lowercase letters or digits. Including the region avoids
# reusing a name that DNS caches may still map to a deleted registry elsewhere.
ACR="${ACR:-ctrdash$(printf %s "$SUBSCRIPTION_ID$LOCATION" | shasum | cut -c1-10)}"

echo "Subscription: $(az account show --query name --output tsv)"
echo "Region: $LOCATION | resource group: $RG | registry: $ACR | GitHub repo: $REPO"

echo "==> Registering resource providers (first run can take a few minutes)"
az extension add --name containerapp --upgrade --only-show-errors
for ns in Microsoft.App Microsoft.OperationalInsights Microsoft.ContainerRegistry Microsoft.ManagedIdentity; do
  az provider register --namespace "$ns" --wait --output none
done

echo "==> Resource group and container registry"
az group create --name "$RG" --location "$LOCATION" --output none
RG_ID="$(az group show --name "$RG" --query id --output tsv)"
az acr create --name "$ACR" --resource-group "$RG" --location "$LOCATION" --sku Basic --output none
ACR_ID="$(az acr show --name "$ACR" --query id --output tsv)"

echo "==> Identity the app pulls images with (AcrPull on the registry only)"
az identity create --name "$APP-pull" --resource-group "$RG" --location "$LOCATION" --output none
PULL_ID="$(az identity show --name "$APP-pull" --resource-group "$RG" --query id --output tsv)"
PULL_PRINCIPAL="$(az identity show --name "$APP-pull" --resource-group "$RG" --query principalId --output tsv)"
az role assignment create --assignee-object-id "$PULL_PRINCIPAL" --assignee-principal-type ServicePrincipal \
  --role AcrPull --scope "$ACR_ID" --output none

echo "==> Identity GitHub Actions signs in as (push images, update the app; this resource group only)"
az identity create --name "$APP-github" --resource-group "$RG" --location "$LOCATION" --output none
GITHUB_CLIENT_ID="$(az identity show --name "$APP-github" --resource-group "$RG" --query clientId --output tsv)"
GITHUB_PRINCIPAL="$(az identity show --name "$APP-github" --resource-group "$RG" --query principalId --output tsv)"
az role assignment create --assignee-object-id "$GITHUB_PRINCIPAL" --assignee-principal-type ServicePrincipal \
  --role AcrPush --scope "$ACR_ID" --output none
az role assignment create --assignee-object-id "$GITHUB_PRINCIPAL" --assignee-principal-type ServicePrincipal \
  --role Contributor --scope "$RG_ID" --output none
# Trust GitHub's OIDC tokens for this repo's main branch (build job) and production environment (deploy job).
for target in "main:ref:refs/heads/main" "production:environment:production"; do
  az identity federated-credential create --identity-name "$APP-github" --resource-group "$RG" \
    --name "github-${target%%:*}" --issuer https://token.actions.githubusercontent.com \
    --subject "$SUBJECT_PREFIX:${target#*:}" --audiences api://AzureADTokenExchange --output none
done

echo "==> Building the first image from $SOURCE_REF (later images come from GitHub Actions)"
git fetch --quiet origin
SHA="$(git rev-parse "$SOURCE_REF")"
SRC="$(mktemp -d)"
git archive "$SHA" | tar -x -C "$SRC"
IMAGE="$ACR.azurecr.io/$IMAGE_NAME:$SHA"
az acr login --name "$ACR"
docker buildx build --platform linux/amd64 --build-arg GIT_SHA="$SHA" --tag "$IMAGE" --push "$SRC"
rm -rf "$SRC"

echo "==> Container Apps environment and app"
# Standard (workload profiles) mode: newer CLIs may otherwise create an "express" environment, which does not
# support the per-commit revision names the pipeline uses.
az containerapp env create --name "$ENVIRONMENT" --resource-group "$RG" --location "$LOCATION" \
  --environment-mode WorkloadProfiles --output none
az containerapp create --name "$APP" --resource-group "$RG" --environment "$ENVIRONMENT" \
  --image "$IMAGE" --registry-server "$ACR.azurecr.io" \
  --user-assigned "$PULL_ID" --registry-identity "$PULL_ID" \
  --ingress external --target-port 8501 \
  --cpu 1.0 --memory 2.0Gi --min-replicas 0 --max-replicas 1 \
  --revision-suffix "git-${SHA:0:7}-setup" --output none
URL="https://$(az containerapp show --name "$APP" --resource-group "$RG" \
  --query properties.configuration.ingress.fqdn --output tsv)"

echo "==> Connecting GitHub Actions"
gh api --method PUT "repos/$REPO/environments/production" --silent
gh secret set AZURE_CLIENT_ID --repo "$REPO" --body "$GITHUB_CLIENT_ID"
gh secret set AZURE_TENANT_ID --repo "$REPO" --body "$TENANT_ID"
gh secret set AZURE_SUBSCRIPTION_ID --repo "$REPO" --body "$SUBSCRIPTION_ID"
gh variable set AZURE_RESOURCE_GROUP --repo "$REPO" --body "$RG"
gh variable set AZURE_ACR_NAME --repo "$REPO" --body "$ACR"
gh variable set AZURE_CONTAINER_APP --repo "$REPO" --body "$APP"
gh variable set AZURE_APP_URL --repo "$REPO" --body "$URL"  # used by the daily health check

echo
echo "Live app: $URL"
echo "From now on every push to main is tested, built and deployed by GitHub Actions."
