# Concrete values for the cpu-rag v2 trial box. No secrets here (the RAG_API_KEY
# lives in the bundle's .env on the instance, not in Terraform).

region            = "eu-central-1"
availability_zone = "eu-central-1a"

# SSH + API open to the internet (our IP is dynamic; client is IP-independent via
# the X-API-Key). SSH is key-only; the box runs only while in use. To lock down:
#   ssh_cidrs = ["90.160.73.35/32"]
#   api_cidrs = ["90.160.73.35/32", "<client>/32"]
ssh_cidrs = ["0.0.0.0/0"]
api_cidrs = ["0.0.0.0/0"]

# r7i.2xlarge = 8 vCPU / 64 GB (SPR) — both profiles + native-image option.
instance_type  = "r7i.2xlarge"
root_volume_gb = 50
