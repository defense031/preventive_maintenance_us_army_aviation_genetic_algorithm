#!/bin/bash
# ====================================================================================
# Aviation GA Cloud Deployment Script
# ====================================================================================
# This script helps deploy and run the GA optimization on AWS EC2
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Docker installed locally
#   - SSH key pair for EC2 access
#
# Usage:
#   ./scripts/deploy_cloud.sh build          # Build Docker image locally
#   ./scripts/deploy_cloud.sh test           # Test locally with 2 generations
#   ./scripts/deploy_cloud.sh push           # Push to ECR
#   ./scripts/deploy_cloud.sh launch         # Launch EC2 instance
#   ./scripts/deploy_cloud.sh results        # Download results from EC2
# ====================================================================================

set -e

# Configuration
IMAGE_NAME="aviation-ga"
ECR_REPO="aviation-ga"
AWS_REGION="us-east-1"
EC2_INSTANCE_TYPE="c7a.16xlarge"  # 64 vCPUs, ~$2.50/hr
EC2_AMI="ami-0c7217cdde317cfec"   # Amazon Linux 2023 (us-east-1)
KEY_NAME="aviation_ga_key"
SECURITY_GROUP="sg-0743de1cbf1cf427d"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${GREEN}======================================================================${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}======================================================================${NC}\n"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# ====================================================================================
# BUILD - Build Docker image locally
# ====================================================================================
build() {
    print_header "Building Docker Image"

    cd "$(dirname "$0")/.."

    echo "Building image: ${IMAGE_NAME}..."
    docker build -t ${IMAGE_NAME} .

    echo -e "\n${GREEN}✅ Build complete!${NC}"
    echo "Image size: $(docker images ${IMAGE_NAME} --format '{{.Size}}')"
}

# ====================================================================================
# TEST - Run quick local test
# ====================================================================================
test_local() {
    print_header "Running Local Test (2 generations)"

    cd "$(dirname "$0")/.."

    # Create test config
    cat > config/ga/cloud_test.yaml << 'EOF'
name: "cloud_test"
description: "Quick test for cloud deployment"
version: "1.0.0"

population:
  size: 10
  elite_count: 2

operators:
  crossover_rate: 0.70
  mutation_rate: 0.30
  blx_alpha: 0.5
  mutation_start_rate: 0.30
  mutation_min_rate: 0.10
  mutation_start_sigma: 0.30
  mutation_min_sigma: 0.10
  mutation_decay_point: 0.75

selection:
  method: "tournament"
  tournament_size: 3

fitness:
  weights:
    mission_success: 0.70
    operational_readiness: 0.15
    flight_hours: 0.15
  baseline_max_flight_hours: 4563.55

evaluation:
  episodes_per_chromosome: 5
  parallel_workers: 4

convergence:
  max_generations: 2
  early_stopping_patience: 10
  improvement_threshold: 0.001

checkpointing:
  enabled: false
  frequency: 10
  save_dir: "results/cloud_test_checkpoints"
  filename_pattern: "checkpoint_gen{generation:03d}.pkl"

logging:
  verbose: true
  log_dir: "results/cloud_test_logs"
  log_frequency: 1
  track_diversity: true
  track_statistics: true

chromosome:
  config_type: "medium"
  tree_depth: 3
  n_features: 5
  n_fleet_features: 7
  n_splits: 7
  n_leaves: 8
  n_buckets: 4
  n_genes: 31

feature_encoder:
  config_path: "config/features/medium_dt.yaml"
  features:
    - observed_rul
    - hours_to_major
    - hours_to_minor
    - da_line_deviation_positive
    - da_line_deviation_negative
  fleet_features:
    - fmc_count
    - nmc_count
    - mean_observed_rul
    - min_observed_rul
    - mean_hours_to_major
    - min_hours_to_major
    - mean_hours_to_minor
    - min_hours_to_minor

simulation:
  config_path: "config/default.yaml"
  seed: 42

output:
  results_dir: "results/cloud_test"
  save_best_chromosome: true
  save_final_population: false
  save_fitness_history: false
  generate_plots: false
EOF

    echo "Running Docker container..."
    docker run --rm \
        -v "$(pwd)/results:/app/results" \
        ${IMAGE_NAME} \
        python scripts/run_ga.py --config config/ga/cloud_test.yaml --no-confirm

    echo -e "\n${GREEN}✅ Local test complete!${NC}"
    echo "Check results in: results/cloud_test/"
}

# ====================================================================================
# PUSH - Push to Amazon ECR
# ====================================================================================
push_ecr() {
    print_header "Pushing to Amazon ECR"

    # Get AWS account ID
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

    echo -e "\n${GREEN}✅ Push complete!${NC}"
    echo "Image URI: ${ECR_URI}:latest"
}

# ====================================================================================
# LAUNCH - Launch EC2 instance
# ====================================================================================
launch_ec2() {
    print_header "Launching EC2 Instance"

    print_warning "This will launch a ${EC2_INSTANCE_TYPE} instance (~\$2.50/hr)"
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi

    # User data script to run on instance startup
    USER_DATA=$(cat << 'USERDATA'
#!/bin/bash
yum update -y
yum install -y docker
systemctl start docker
systemctl enable docker

# Login to ECR and pull image
AWS_ACCOUNT_ID=$(curl -s http://169.254.169.254/latest/meta-data/identity-credentials/ec2/info | grep AccountId | cut -d'"' -f4)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com

# Pull and run
docker pull ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/aviation-ga:latest
mkdir -p /home/ec2-user/results
docker run -d \
    -v /home/ec2-user/results:/app/results \
    ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/aviation-ga:latest

echo "GA optimization started at $(date)" > /home/ec2-user/run.log
USERDATA
)

    echo "Launching instance..."
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id ${EC2_AMI} \
        --instance-type ${EC2_INSTANCE_TYPE} \
        --key-name ${KEY_NAME} \
        --security-group-ids ${SECURITY_GROUP} \
        --user-data "${USER_DATA}" \
        --iam-instance-profile Name=EC2-ECR-Access \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=aviation-ga-runner}]" \
        --query 'Instances[0].InstanceId' \
        --output text)

    echo "Instance ID: ${INSTANCE_ID}"
    echo "Waiting for instance to start..."

    aws ec2 wait instance-running --instance-ids ${INSTANCE_ID}

    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids ${INSTANCE_ID} \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)

    echo -e "\n${GREEN}✅ Instance launched!${NC}"
    echo ""
    echo "Instance ID:  ${INSTANCE_ID}"
    echo "Public IP:    ${PUBLIC_IP}"
    echo ""
    echo "To SSH:       ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP}"
    echo "To monitor:   ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP} 'docker logs -f \$(docker ps -q)'"
    echo "To download:  scp -i ~/.ssh/${KEY_NAME}.pem -r ec2-user@${PUBLIC_IP}:~/results ./results-cloud/"
    echo ""
    print_warning "Remember to terminate the instance when done!"
    echo "aws ec2 terminate-instances --instance-ids ${INSTANCE_ID}"
}

# ====================================================================================
# RESULTS - Download results from EC2
# ====================================================================================
download_results() {
    print_header "Downloading Results from EC2"

    if [ -z "$2" ]; then
        print_error "Please provide the EC2 public IP"
        echo "Usage: $0 results <EC2_PUBLIC_IP>"
        exit 1
    fi

    EC2_IP=$2

    echo "Downloading from ${EC2_IP}..."
    mkdir -p results-cloud
    scp -i ~/.ssh/${KEY_NAME}.pem -r ec2-user@${EC2_IP}:~/results/* ./results-cloud/

    echo -e "\n${GREEN}✅ Download complete!${NC}"
    echo "Results saved to: results-cloud/"
}

# ====================================================================================
# HELP
# ====================================================================================
show_help() {
    echo "Aviation GA Cloud Deployment Script"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  build     Build Docker image locally"
    echo "  test      Run quick local test (2 generations)"
    echo "  push      Push image to Amazon ECR"
    echo "  launch    Launch EC2 instance and start optimization"
    echo "  results   Download results from EC2 (requires IP argument)"
    echo ""
    echo "Typical workflow:"
    echo "  1. $0 build"
    echo "  2. $0 test"
    echo "  3. $0 push"
    echo "  4. $0 launch"
    echo "  5. $0 results <EC2_IP>"
}

# ====================================================================================
# MAIN
# ====================================================================================
case "${1:-help}" in
    build)
        build
        ;;
    test)
        test_local
        ;;
    push)
        push_ecr
        ;;
    launch)
        launch_ec2
        ;;
    results)
        download_results "$@"
        ;;
    *)
        show_help
        ;;
esac
