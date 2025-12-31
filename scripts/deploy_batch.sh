#!/bin/bash
# ====================================================================================
# Batch Experiment Cloud Deployment Script (Spot Instances)
# ====================================================================================
# Deploy and run multiple GA experiments on large spot instances
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Docker installed locally
#   - SSH key pair for EC2 access
#
# Usage:
#   ./scripts/deploy_batch.sh build                    # Build Docker image
#   ./scripts/deploy_batch.sh push                     # Push to ECR
#   ./scripts/deploy_batch.sh launch-spot              # Launch spot instance
#   ./scripts/deploy_batch.sh launch-spot --size 192   # Launch with 192 vCPUs
#   ./scripts/deploy_batch.sh launch-spot --size 384   # Launch 2x instances (384 vCPUs)
#   ./scripts/deploy_batch.sh status                   # Check running instances
#   ./scripts/deploy_batch.sh results <IP>             # Download results
#   ./scripts/deploy_batch.sh terminate                # Terminate all instances
# ====================================================================================

set -e

# Configuration - UPDATE THESE
IMAGE_NAME="aviation-ga-batch"
ECR_REPO="aviation-ga-batch"
AWS_REGION="us-east-1"
KEY_NAME="aviation_ga_key"
SECURITY_GROUP="sg-0743de1cbf1cf427d"
IAM_ROLE="EC2-ECR-Access"          # IAM instance profile with ECR access

# Instance configurations (function to avoid bash 4+ requirement)
get_instance_type() {
    case "$1" in
        32)  echo "c6i.8xlarge" ;;    # 32 vCPUs, Intel - better spot availability
        48)  echo "c6i.12xlarge" ;;   # 48 vCPUs
        96)  echo "c6i.24xlarge" ;;   # 96 vCPUs
        192) echo "c6i.metal" ;;      # 128 vCPUs (largest c6i)
        *)   echo "" ;;
    esac
}

get_spot_max_price() {
    case "$1" in
        32)  echo "0.90" ;;   # Raised from 0.50 - spot prices fluctuate
        48)  echo "1.00" ;;
        96)  echo "2.00" ;;
        192) echo "4.00" ;;
        *)   echo "1.50" ;;
    esac
}

# Amazon Linux 2023 AMI (us-east-1) - Update for your region
EC2_AMI="ami-0c7217cdde317cfec"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "\n${GREEN}======================================================================${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}======================================================================${NC}\n"
}

print_info() {
    echo -e "${BLUE}INFO: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

# ====================================================================================
# BUILD - Build Docker image with experiments
# ====================================================================================
build() {
    print_header "Building Batch Docker Image"

    cd "$(dirname "$0")/.."

    # Create a batch-optimized Dockerfile
    cat > Dockerfile.batch << 'DOCKERFILE'
# ====================================================================================
# Dockerfile for Aviation GA Batch Optimization (Cloud)
# ====================================================================================
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Copy application code
COPY config/ ./config/
COPY experiments/ ./experiments/
COPY optimization/ ./optimization/
COPY policy/ ./policy/
COPY simulation/ ./simulation/
COPY utils/ ./utils/
COPY scripts/ ./scripts/

# Create results directory
RUN mkdir -p results

# Set environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default: run batch experiments
# Override with: docker run ... python scripts/run_batch_experiments.py --experiments "..."
CMD ["python", "scripts/run_batch_experiments.py", "--help"]
DOCKERFILE

    echo "Building image: ${IMAGE_NAME}..."
    docker build -f Dockerfile.batch -t ${IMAGE_NAME} .

    echo -e "\n${GREEN}Build complete!${NC}"
    echo "Image size: $(docker images ${IMAGE_NAME} --format '{{.Size}}')"
}

# ====================================================================================
# PUSH - Push to ECR
# ====================================================================================
push_ecr() {
    print_header "Pushing to Amazon ECR"

    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

    echo "Creating ECR repository (if not exists)..."
    aws ecr create-repository --repository-name ${ECR_REPO} --region ${AWS_REGION} 2>/dev/null || true

    echo "Authenticating Docker to ECR..."
    aws ecr get-login-password --region ${AWS_REGION} | \
        docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

    echo "Tagging image..."
    docker tag ${IMAGE_NAME}:latest ${ECR_URI}:latest

    echo "Pushing to ECR..."
    docker push ${ECR_URI}:latest

    echo -e "\n${GREEN}Push complete!${NC}"
    echo "Image URI: ${ECR_URI}:latest"
}

# ====================================================================================
# LAUNCH-SPOT - Launch spot instance(s)
# ====================================================================================
launch_spot() {
    print_header "Launching Spot Instance"

    # Parse arguments
    local VCPU_SIZE="192"  # Default to 192 vCPUs
    local EXPERIMENT_PATTERN="experiments/ga_configs/exp_simple_*_ms70*.yaml"
    local WORKERS_PER_EXP="12"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --size)
                VCPU_SIZE="$2"
                shift 2
                ;;
            --experiments)
                EXPERIMENT_PATTERN="$2"
                shift 2
                ;;
            --workers)
                WORKERS_PER_EXP="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done

    # Validate size and get instance type
    local INSTANCE_TYPE=$(get_instance_type "$VCPU_SIZE")
    if [[ -z "$INSTANCE_TYPE" ]]; then
        print_error "Invalid size: $VCPU_SIZE. Valid options: 48, 96, 192"
        exit 1
    fi

    local MAX_PRICE=$(get_spot_max_price "$VCPU_SIZE")

    # Get AWS account ID
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

    echo "Instance Configuration:"
    echo "  Type:         ${INSTANCE_TYPE} (${VCPU_SIZE} vCPUs)"
    echo "  Spot Max:     \$${MAX_PRICE}/hr"
    echo "  Experiments:  ${EXPERIMENT_PATTERN}"
    echo "  Workers/Exp:  ${WORKERS_PER_EXP}"
    echo ""

    # Check current spot price
    CURRENT_SPOT=$(aws ec2 describe-spot-price-history \
        --instance-types ${INSTANCE_TYPE} \
        --product-descriptions "Linux/UNIX" \
        --max-items 1 \
        --query 'SpotPriceHistory[0].SpotPrice' \
        --output text 2>/dev/null || echo "unknown")

    echo "  Current Spot: \$${CURRENT_SPOT}/hr"
    echo ""

    read -p "Launch spot instance? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi

    # User data script for spot instance
    USER_DATA=$(cat << USERDATA
#!/bin/bash
set -e

# Log everything
exec > >(tee /var/log/user-data.log) 2>&1
echo "Starting setup at \$(date)"

# Install Docker
yum update -y
yum install -y docker
systemctl start docker
systemctl enable docker

# ECR login
AWS_ACCOUNT_ID=\$(curl -s http://169.254.169.254/latest/meta-data/identity-credentials/ec2/info | grep AccountId | cut -d'"' -f4)
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin \${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Pull image
echo "Pulling image..."
docker pull ${ECR_URI}:latest

# Create results directory
mkdir -p /home/ec2-user/results
chown ec2-user:ec2-user /home/ec2-user/results

# Run batch experiments
echo "Starting batch experiments at \$(date)"
docker run -d --name ga-batch \
    -v /home/ec2-user/results:/app/results \
    ${ECR_URI}:latest \
    python scripts/run_batch_experiments.py \
        --experiments "${EXPERIMENT_PATTERN}" \
        --workers-per-exp ${WORKERS_PER_EXP} \
        --no-confirm

# Write status file
echo "Launched at \$(date)" > /home/ec2-user/status.txt
echo "Experiments: ${EXPERIMENT_PATTERN}" >> /home/ec2-user/status.txt
echo "Workers per experiment: ${WORKERS_PER_EXP}" >> /home/ec2-user/status.txt
USERDATA
)

    echo "Requesting spot instance..."

    # Create spot instance request
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id ${EC2_AMI} \
        --instance-type ${INSTANCE_TYPE} \
        --key-name ${KEY_NAME} \
        --security-group-ids ${SECURITY_GROUP} \
        --user-data "${USER_DATA}" \
        --iam-instance-profile Name=${IAM_ROLE} \
        --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"'"${MAX_PRICE}"'","SpotInstanceType":"one-time"}}' \
        --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ga-batch-${VCPU_SIZE}vcpu},{Key=Project,Value=aviation-ga}]" \
        --query 'Instances[0].InstanceId' \
        --output text)

    echo "Instance ID: ${INSTANCE_ID}"
    echo "Waiting for instance to start..."

    aws ec2 wait instance-running --instance-ids ${INSTANCE_ID}

    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids ${INSTANCE_ID} \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)

    echo -e "\n${GREEN}Spot instance launched!${NC}"
    echo ""
    echo "Instance ID:     ${INSTANCE_ID}"
    echo "Public IP:       ${PUBLIC_IP}"
    echo "Instance Type:   ${INSTANCE_TYPE} (${VCPU_SIZE} vCPUs)"
    echo ""
    echo "Commands:"
    echo "  SSH:           ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP}"
    echo "  Monitor:       ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP} 'docker logs -f ga-batch'"
    echo "  Status:        ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP} 'docker ps'"
    echo "  Download:      ./scripts/deploy_batch.sh results ${PUBLIC_IP}"
    echo ""
    print_warning "REMEMBER: Terminate when done to avoid charges!"
    echo "  Terminate:     aws ec2 terminate-instances --instance-ids ${INSTANCE_ID}"
}

# ====================================================================================
# STATUS - Check running instances
# ====================================================================================
status() {
    print_header "Running GA Batch Instances"

    aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=aviation-ga" "Name=instance-state-name,Values=running,pending" \
        --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,PublicIpAddress,LaunchTime,Tags[?Key==`Name`].Value|[0]]' \
        --output table
}

# ====================================================================================
# RESULTS - Download results
# ====================================================================================
results() {
    print_header "Downloading Results"

    if [ -z "$1" ]; then
        print_error "Please provide the EC2 public IP"
        echo "Usage: $0 results <EC2_PUBLIC_IP>"
        exit 1
    fi

    EC2_IP=$1
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOCAL_DIR="results-cloud-${TIMESTAMP}"

    echo "Downloading from ${EC2_IP}..."
    mkdir -p ${LOCAL_DIR}
    scp -i ~/.ssh/${KEY_NAME}.pem -r ec2-user@${EC2_IP}:~/results/* ./${LOCAL_DIR}/

    echo -e "\n${GREEN}Download complete!${NC}"
    echo "Results saved to: ${LOCAL_DIR}/"
    ls -la ${LOCAL_DIR}/
}

# ====================================================================================
# TERMINATE - Terminate all GA instances
# ====================================================================================
terminate_all() {
    print_header "Terminating GA Instances"

    INSTANCE_IDS=$(aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=aviation-ga" "Name=instance-state-name,Values=running,pending" \
        --query 'Reservations[*].Instances[*].InstanceId' \
        --output text)

    if [ -z "$INSTANCE_IDS" ]; then
        echo "No running instances found."
        exit 0
    fi

    echo "Found instances: ${INSTANCE_IDS}"
    read -p "Terminate these instances? (y/n) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        aws ec2 terminate-instances --instance-ids ${INSTANCE_IDS}
        echo -e "\n${GREEN}Termination initiated!${NC}"
    else
        echo "Aborted."
    fi
}

# ====================================================================================
# ESTIMATE - Estimate cost and runtime
# ====================================================================================
estimate() {
    print_header "Cost & Runtime Estimates"

    echo "Available instance configurations:"
    echo ""
    printf "%-8s %-16s %-12s %-12s %-15s\n" "vCPUs" "Instance" "Spot/hr" "On-Demand" "Experiments"
    printf "%-8s %-16s %-12s %-12s %-15s\n" "-----" "--------" "-------" "---------" "-----------"
    printf "%-8s %-16s %-12s %-12s %-15s\n" "48" "c7a.12xlarge" "~\$0.55" "\$1.85" "~4 parallel"
    printf "%-8s %-16s %-12s %-12s %-15s\n" "96" "c7a.24xlarge" "~\$1.10" "\$3.70" "~8 parallel"
    printf "%-8s %-16s %-12s %-12s %-15s\n" "192" "c7a.48xlarge" "~\$2.20" "\$7.34" "~16 parallel"
    echo ""
    echo "Estimated runtime per experiment: ~30-60 minutes (with early stopping)"
    echo ""
    echo "Example: 15 experiments on c7a.48xlarge (192 vCPUs)"
    echo "  - Experiments run: ~15 parallel (12 workers each = 180 CPUs used)"
    echo "  - Estimated time: ~45 minutes"
    echo "  - Estimated cost: 0.75 hrs × \$2.20 = ~\$1.65 (spot)"
}

# ====================================================================================
# HELP
# ====================================================================================
show_help() {
    echo "Aviation GA Batch Deployment Script (Spot Instances)"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  build                      Build Docker image with experiments"
    echo "  push                       Push image to Amazon ECR"
    echo "  launch-spot [options]      Launch spot instance"
    echo "    --size <48|96|192>       vCPU count (default: 192)"
    echo "    --experiments <pattern>  Glob pattern for experiments"
    echo "    --workers <n>            Workers per experiment (default: 12)"
    echo "  status                     Show running instances"
    echo "  results <IP>               Download results from instance"
    echo "  terminate                  Terminate all GA instances"
    echo "  estimate                   Show cost/runtime estimates"
    echo ""
    echo "Example workflow:"
    echo "  1. $0 build"
    echo "  2. $0 push"
    echo "  3. $0 launch-spot --size 192 --experiments 'experiments/ga_configs/exp_simple_*_ms70.yaml'"
    echo "  4. $0 status"
    echo "  5. $0 results <IP>"
    echo "  6. $0 terminate"
}

# ====================================================================================
# MAIN
# ====================================================================================
case "${1:-help}" in
    build)
        build
        ;;
    push)
        push_ecr
        ;;
    launch-spot)
        shift
        launch_spot "$@"
        ;;
    status)
        status
        ;;
    results)
        results "$2"
        ;;
    terminate)
        terminate_all
        ;;
    estimate)
        estimate
        ;;
    *)
        show_help
        ;;
esac
