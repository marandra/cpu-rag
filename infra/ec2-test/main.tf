# ---------------------------------------------------------------------------
# Dedicated VPC — isolates this internet-facing box from the ~20 other corporate
# instances in the shared account. VPC/subnet/IGW/route table are all FREE (only
# NAT costs money, and we don't use one — public subnet + IGW).
#
# CIDR 10.30.0.0/16 chosen to overlap NOTHING existing: not the parallelcluster
# PoC (10.20.0.0/16), not the default/other VPCs (172.31/34/35/40/60, 10.209,
# 192.168). Same isolation posture we already validated for parallelcluster.
# ---------------------------------------------------------------------------
resource "aws_vpc" "rag" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = { Name = "${var.name}-vpc" }
}

resource "aws_internet_gateway" "rag" {
  vpc_id = aws_vpc.rag.id
  tags   = { Name = "${var.name}-igw" }
}

resource "aws_subnet" "rag" {
  vpc_id                  = aws_vpc.rag.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 4, 0) # 10.30.0.0/20
  availability_zone       = var.availability_zone
  map_public_ip_on_launch = true
  tags                    = { Name = "${var.name}-public-${var.availability_zone}" }
}

resource "aws_route_table" "rag" {
  vpc_id = aws_vpc.rag.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.rag.id
  }
  tags = { Name = "${var.name}-public-rt" }
}

resource "aws_route_table_association" "rag" {
  subnet_id      = aws_subnet.rag.id
  route_table_id = aws_route_table.rag.id
}

# ---------------------------------------------------------------------------
# Latest Ubuntu 24.04 (Noble) AMI from Canonical.
# ---------------------------------------------------------------------------
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# ---------------------------------------------------------------------------
# Dedicated key pair (isolates this project's SSH access in a shared account).
# ---------------------------------------------------------------------------
resource "aws_key_pair" "rag" {
  key_name   = "${var.name}-key"
  public_key = file(pathexpand(var.ssh_public_key_path))
}

# ---------------------------------------------------------------------------
# Security group: SSH from us only; API ports from us + the client.
# ---------------------------------------------------------------------------
resource "aws_security_group" "rag" {
  name        = "${var.name}-sg"
  description = "cpu-rag v2 trial: SSH (us) + API 8001/8002 (us + client)"
  vpc_id      = aws_vpc.rag.id

  ingress {
    description = "SSH (key-only; open by default because our IP is dynamic). SSM is the no-exposure alternative."
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.ssh_cidrs
  }

  ingress {
    description = "RAG API ports (glucowise 8001, aiciblock 8002); auth is the X-API-Key header."
    from_port   = 8001
    to_port     = 8002
    protocol    = "tcp"
    cidr_blocks = var.api_cidrs
  }

  egress {
    description = "All outbound (model download from huggingface.co, apt, SSM endpoints)."
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.name}-sg" }
}

# ---------------------------------------------------------------------------
# The instance.
# ---------------------------------------------------------------------------
resource "aws_instance" "rag" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.rag.key_name
  vpc_security_group_ids = [aws_security_group.rag.id]
  subnet_id              = aws_subnet.rag.id
  iam_instance_profile   = aws_iam_instance_profile.rag.name
  user_data              = file("${path.module}/user-data.sh")

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_gb
    delete_on_termination = true
    tags                  = { Name = "${var.name}-root" }
  }

  # Enforce IMDSv2.
  metadata_options {
    http_tokens   = "required"
    http_endpoint = "enabled"
  }

  tags = { Name = var.name }
}

# ---------------------------------------------------------------------------
# Elastic IP: a stable address to hand the client that survives stop/start.
# ---------------------------------------------------------------------------
resource "aws_eip" "rag" {
  instance = aws_instance.rag.id
  domain   = "vpc"
  tags     = { Name = "${var.name}-eip" }
}
