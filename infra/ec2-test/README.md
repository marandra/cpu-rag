# cpu-rag v2 trial EC2 — OpenTofu module

Provisions **our own** EC2 to run the v2 client bundle (both profiles) for a
trial/repro box. Spec: [`../../docs/ec2_test_env.md`](../../docs/ec2_test_env.md).

This runs in a **shared corporate AWS account** (`092768957966`, ~20 other
instances). Everything here is built to coexist safely — same isolation posture
we validated for the `parallelcluster` PoC:

- **Dedicated VPC** `10.30.0.0/16` (overlaps nothing existing) — the box does not
  sit in the shared default VPC.
- **`default_tags`** `Project=cpu-rag Owner=marcelo Env=test ManagedBy=opentofu`
  on every resource → identifiable and destroyable without touching anything else.
- **Least-privilege IAM**: instance role has *only* `AmazonSSMManagedInstanceCore`.
- **SG**: SSH (22, key-only) + API (8001–8002) open to the internet by default —
  our IP is dynamic and the client is IP-independent via the `X-API-Key`. Both are
  `*_cidrs` variables, so tightening to a /32 list is a one-line change.

## What it creates

Dedicated VPC + public subnet + IGW, a key pair, a security group, an SSM
instance role, one `r7i.2xlarge` (Ubuntu 24.04, 50 GB gp3, Docker via user-data),
and an Elastic IP. State is **local** (`terraform.tfstate`, gitignored) — only
this workstation can `destroy` it, so **whoever applies owns the teardown**.

## Use

```bash
cd infra/ec2-test
tofu init
tofu plan          # review
tofu apply         # creates the box (~$0.53/h while running)

# after apply:
tofu output ssh_command      # ssh -i ~/.ssh/id_rag_ec2 ubuntu@<eip>
tofu output ssm_command      # shell without SSH, if our IP changes
tofu output api_urls
```

Then follow [`docs/ec2_test_env.md`](../../docs/ec2_test_env.md) §5 to scp the
bundle and bring both profiles up.

### Lock down the SG

Edit `terraform.tfvars` (`ssh_cidrs`, `api_cidrs`) to a /32 list then `tofu
apply` — only the SG changes.

## Start / stop (day-to-day)

Use the helper — it reads the instance id from tofu state (no hardcoded IDs) and
the Elastic IP is stable, so `ssh rag` keeps working across stop/start:

```bash
./rag-ec2.sh start    # power on, wait, show IP + health
./rag-ec2.sh stop     # power off — kills the ~$0.53/h compute charge
./rag-ec2.sh status   # state + IP
./rag-ec2.sh ssh      # ssh in (once running)
```

Stopped, only EBS+EIP remain (~$0.25/day). Full `$0` needs `tofu destroy` (below),
which also drops the 17 GB model and changes the IP — so keep it for end-of-trial.

## Teardown (the "$0 when idle" button)

```bash
# Stop when idle — keeps only the EBS + EIP charge (a few cents/day):
aws ec2 stop-instances --region eu-central-1 --instance-ids $(tofu output -raw instance_id)

# Destroy when the trial is over — removes everything this module made:
tofu destroy
```
