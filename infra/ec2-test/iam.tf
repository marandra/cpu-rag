# ---------------------------------------------------------------------------
# SSM instance profile — least-privilege admin shell WITHOUT exposing anything.
# AmazonSSMManagedInstanceCore is the only policy: it lets Session Manager open a
# shell (and nothing else — no S3, no EC2 API). SSH on :22 stays open to our /32
# purely so `scp` can push the bundle; SSM is the fallback if the IP changes.
#
#   aws ssm start-session --target <instance-id>
# ---------------------------------------------------------------------------
data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "rag" {
  name               = "${var.name}-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.rag.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "rag" {
  name = "${var.name}-profile"
  role = aws_iam_role.rag.name
}
