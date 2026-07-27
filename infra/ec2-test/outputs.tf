output "public_ip" {
  description = "Elastic IP — the stable address to share (SSH + API)."
  value       = aws_eip.rag.public_ip
}

output "instance_id" {
  value = aws_instance.rag.id
}

output "region" {
  value = var.region
}

output "ssh_command" {
  description = "Ready-to-paste SSH command."
  value       = "ssh -i ~/.ssh/id_rag_ec2 ubuntu@${aws_eip.rag.public_ip}"
}

output "api_urls" {
  description = "Health endpoints once the bundle is up."
  value = {
    glucowise = "http://${aws_eip.rag.public_ip}:8001/health"
    aiciblock = "http://${aws_eip.rag.public_ip}:8002/health"
  }
}

output "ami_id" {
  value = data.aws_ami.ubuntu.id
}

output "ssm_command" {
  description = "Open a shell without SSH (fallback if our IP changes)."
  value       = "aws ssm start-session --region ${var.region} --target ${aws_instance.rag.id}"
}
