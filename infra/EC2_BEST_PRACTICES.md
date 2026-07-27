# EC2 provisioning — best-practices (DRAFT)

Distilled from the `cpu-rag` v2 trial box and the `parallelcluster` PoC, both in
the **shared** Eurecat AWS account. Draft — to be moved to a dedicated infra
folder. The goal: spin up a throwaway/trial EC2 that coexists safely with ~20
other projects' resources, is cheap, and is trivial to tear down.

---

## 0. Golden rules (the short version)

1. **Tag everything** (`Project`, `Owner`, `Env`, `ManagedBy`) via provider
   `default_tags` — so anyone can tell what's yours and delete it safely.
2. **Dedicated VPC, non-overlapping CIDR** — never drop an internet-facing box in
   the shared default VPC.
3. **Least privilege** — no IAM role unless needed; if you need a shell, use SSM,
   not a broad role. SSH is **key-only**.
4. **Elastic IP + stop/start**, not destroy/recreate, for a stable address.
5. **Stop when idle, destroy when done.** Whoever `apply`s owns the teardown.
6. **IaC or it didn't happen** — OpenTofu, state gitignored, reproducible.

---

## 1. Shared-account hygiene

The account (`092768957966`, eu-central-1) is **not a sandbox**: many running
instances and key pairs from other projects. Therefore:

- **`default_tags` on the provider**, applied to every resource:
  ```hcl
  provider "aws" {
    region = var.region
    default_tags { tags = {
      Project = "<proj>", Owner = "<you>", Env = "test", ManagedBy = "opentofu"
    } }
  }
  ```
- **Prefix every resource name** with the project (`${var.name}-sg`, `-vpc`, …) so
  it never collides with another stack and is greppable in the console.
- **Dedicated key pair** per project (`ssh-keygen -t ed25519 -f ~/.ssh/id_<proj>`),
  imported via `aws_key_pair`. Don't reuse another project's key.
- **Check service quotas** before consuming shared capacity (On-Demand vCPU by
  family, EIP count, VPC count). A few extra vCPU is usually fine; note it.

## 2. Network — dedicated VPC

VPC / subnet / IGW / route table are **free** (only NAT gateways cost money — a
public subnet + IGW needs no NAT). So isolation is free:

- One dedicated `aws_vpc` with a CIDR that overlaps **nothing** existing. Known
  in-use ranges to avoid: `172.31/34/35/40/60`, `10.209`, `192.168`, plus other
  projects' dedicated VPCs (`parallelcluster` = `10.20.0.0/16`, `cpu-rag` =
  `10.30.0.0/16`). **Pick the next free /16** and record it here.
- One public subnet + IGW + a `0.0.0.0/0` route; `map_public_ip_on_launch = true`.
- Enforce **IMDSv2** on the instance (`metadata_options { http_tokens = "required" }`).

## 3. Access & security

- **SSH is key-only.** Ubuntu/AL2023 cloud images disable password auth, so an
  open `:22` is scan noise, not a realistic compromise vector. Restrict to a /32
  when your source IP is static; open `0.0.0.0/0` is acceptable for a
  dynamic-IP box **that is stopped when idle**.
- **SSM Session Manager** as the no-exposure admin path: attach an instance role
  with **only** `AmazonSSMManagedInstanceCore`. Gives `aws ssm start-session`
  regardless of source IP — the safety net if you ever lock the SG down.
- **Security group** = the real boundary. Only the ports you serve, sourced from
  the narrowest CIDR that still works. Public API ports are fine **iff** the app
  authenticates (e.g. an `X-API-Key` header) — but remember plain HTTP is
  cleartext; use a strong key and treat TLS as a follow-up for anything beyond a
  short trial.
- **No secrets in Terraform.** App secrets (API keys, `.env`) live on the box,
  not in state or tfvars.

## 4. Sizing & cost

- **Find the binding constraint first.** For LLM/RAG boxes it's usually **RAM**
  (model resident in memory), not vCPU — size on that, use a memory-optimized
  family (`r7i`) rather than paying for cores you won't use.
- **SPR (Sapphire Rapids: m7i/c7i/r7i)** if you need AVX-512/AMX (e.g. a
  native-ISA build). Otherwise any modern family runs a portable image.
- **Stop when idle** (`aws ec2 stop-instances …`) — you keep only the EBS (+EIP)
  charge, cents/day. **Destroy** (`tofu destroy`) at true end-of-trial.
- Think in **single-digit dollars** for a few-hour trial; don't bother with
  reserved/spot for something this short-lived.

## 5. State & tooling

- **OpenTofu** (`tofu`), providers pinned (`~> 5.60`). Commit `.terraform.lock.hcl`;
  **gitignore** `.terraform/`, `*.tfstate*`, `*.auto.tfvars`.
- **Local state = single-operator.** Only the workstation holding
  `terraform.tfstate` can `destroy`. For anything multi-operator or long-lived,
  use an **S3 backend + DynamoDB lock** so the team can manage it and state can't
  be lost. Until then: **whoever applies owns the teardown.**
- Set a shared **`TF_PLUGIN_CACHE_DIR`** (`~/.terraform.d/plugin-cache`) so each
  project doesn't re-download the ~500 MB AWS provider.

## 6. Lifecycle & the stable-IP question

The address model has **two very different cases** — know which one you're in:

| Action | IP behaviour | What to update |
|---|---|---|
| **stop → start** (idle flow) | **EIP unchanged** | **Nothing.** This is why we use an EIP. |
| **`tofu destroy` → `apply`** | EIP is released & a **new** one allocated | SSH config + tell the client |

So for day-to-day, **stop/start — never destroy** — and the IP, the client's URL,
and your `~/.ssh/config` all stay valid with zero effort.

If you genuinely need the **same IP to survive a full destroy/recreate**, don't
let Terraform destroy the EIP:

- **Reserve the EIP out-of-band** (`aws ec2 allocate-address`) and have the module
  *associate* (not allocate) it — or keep the EIP in a **separate, long-lived tofu
  state** from the instance. An idle unassociated EIP costs ~$0.005/h (~$3.6/mo).
- Or put a **DNS name in front** (Route53 A-record → EIP). The client uses a
  hostname that never changes even if the IP does; you just repoint the record.
  This is the right move once a client depends on the endpoint.

### Updating your local SSH config when the IP does change

Keep the host in its own `~/.ssh/config.d/` file and sync it from tofu output:

```bash
IP=$(tofu -chdir=infra/<proj> output -raw public_ip)
sed -i "s/^\( *HostName *\).*/\1$IP/" ~/.ssh/config.d/30-<proj>.conf
```

### Telling the client

There's no auto-discovery for them. Either (a) keep the EIP stable (they keep the
same URL), or (b) front it with a DNS name and hand them the hostname once. Only
notify them of a raw IP change if you have neither.

## 7. Teardown checklist

- [ ] `tofu destroy` (removes VPC, instance, EIP, SG, IAM role, key pair).
- [ ] Confirm no leftover **EBS volumes** / **EIPs** (they bill even when idle) —
      `aws ec2 describe-volumes` / `describe-addresses` filtered by your tags.
- [ ] Remove the `~/.ssh/config.d/` entry if the box is gone for good.

---

### Reusable skeleton

See `infra/ec2-test/` in `cpu-rag` for a working reference: `versions.tf`
(provider + default_tags), `main.tf` (VPC/subnet/IGW/SG/instance/EIP), `iam.tf`
(SSM role), `variables.tf` + `terraform.tfvars`, `user-data.sh`, `outputs.tf`,
`README.md` (teardown button).
