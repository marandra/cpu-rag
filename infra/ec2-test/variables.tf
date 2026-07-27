variable "region" {
  description = "AWS region. Account default is eu-central-1 (Frankfurt)."
  type        = string
  default     = "eu-central-1"
}

variable "availability_zone" {
  description = "AZ for the public subnet."
  type        = string
  default     = "eu-central-1a"
}

variable "vpc_cidr" {
  description = "CIDR for the DEDICATED VPC. 10.30/16 overlaps nothing existing (pcluster is 10.20/16; defaults are 172.31/34/35/40/60, 10.209, 192.168)."
  type        = string
  default     = "10.30.0.0/16"
}

variable "instance_type" {
  description = "SPR instance. r7i.2xlarge = 8 vCPU / 64 GB, runs both profiles + keeps the native-image option."
  type        = string
  default     = "r7i.2xlarge"
}

variable "root_volume_gb" {
  description = "Root gp3 size. 17 GB model + snapshots + image + OS."
  type        = number
  default     = 50
}

variable "ssh_cidrs" {
  description = <<-EOT
    CIDRs allowed on SSH (22). Default OPEN — our source IP is dynamic. Safe
    because Ubuntu cloud images are key-only (no password auth), so an open 22 is
    scan noise, not a realistic compromise vector; stopping the instance when idle
    removes even that. Tighten to a /32 list here if we ever want to.
  EOT
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "api_cidrs" {
  description = <<-EOT
    CIDRs allowed on the API ports (8001-8002). Default OPEN — the client connects
    with the X-API-Key header, IP-independent. Auth IS the key; note the trial runs
    plain HTTP (TLS out of scope), so use a strong RAG_API_KEY.
  EOT
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "ssh_public_key_path" {
  description = "Path to the dedicated public key imported as the instance key pair."
  type        = string
  default     = "~/.ssh/id_rag_ec2.pub"
}

variable "name" {
  description = "Name tag / prefix for the resources."
  type        = string
  default     = "cpu-rag-ec2-test"
}
