terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }
}

provider "aws" {
  region = var.region
  # Credentials come from the environment (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
  # or ~/.aws. This account is SHARED corporate infra — every resource below is
  # tagged so it can be identified and destroyed without touching anything else.
  default_tags {
    tags = {
      Project   = "cpu-rag"
      Owner     = "marcelo"
      Env       = "test"
      ManagedBy = "opentofu"
      Purpose   = "v2-deliverable-trial"
    }
  }
}
