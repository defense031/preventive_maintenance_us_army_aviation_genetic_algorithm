#!/bin/bash
# run_fleet_analysis.sh
# Convenient wrapper script for generating fleet dashboards
#
# USAGE:
#   ./scripts/run_fleet_analysis.sh                    # Use most recent results
#   ./scripts/run_fleet_analysis.sh <results_dir>      # Specify results directory
#   ./scripts/run_fleet_analysis.sh --help             # Show help

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Help message
show_help() {
    cat << EOF
Fleet Analysis Dashboard Generator

USAGE:
    $(basename "$0") [options] [results_dir]

ARGUMENTS:
    results_dir             Path to results directory containing simulation_data.db
                           (defaults to most recent in results/)

OPTIONS:
    -h, --help             Show this help message
    -s, --session NAME     Session name for plot titles
    -e, --episode ID       Episode ID to visualize (defaults to last episode)

EXAMPLES:
    # Analyze most recent results
    $(basename "$0")

    # Analyze specific results directory
    $(basename "$0") results/test_validation_20251119_231610_2ep_seed42

    # With custom session name
    $(basename "$0") -s "Baseline Run" results/baseline_*/

    # Specific episode
    $(basename "$0") -e 1 results/test_*/

OUTPUT:
    Saves plots to: {results_dir}/visualizations/
    - fleet_dashboard_or_trend.png
    - fleet_dashboard_fleet_rul.png
    - fleet_dashboard_maintenance_timeline.png
    - fleet_dashboard_performance_analysis.png
    - fleet_dashboard_combined.png

EOF
}

# Parse arguments
SESSION_NAME=""
EPISODE_ID=""
RESULTS_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--session)
            SESSION_NAME="$2"
            shift 2
            ;;
        -e|--episode)
            EPISODE_ID="$2"
            shift 2
            ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
        *)
            RESULTS_DIR="$1"
            shift
            ;;
    esac
done

# Find results directory if not specified
if [ -z "$RESULTS_DIR" ]; then
    echo -e "${BLUE}🔍 Finding most recent results directory...${NC}"

    # Look for most recent results directory containing simulation_data.db
    RESULTS_DIR=$(find results -name "simulation_data.db" -type f 2>/dev/null | \
                  head -1 | xargs -I {} dirname {})

    if [ -z "$RESULTS_DIR" ]; then
        echo -e "${RED}❌ No results directories found in results/${NC}"
        echo -e "${YELLOW}💡 Make sure you've run a simulation first${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ Found: $RESULTS_DIR${NC}"
fi

# Verify results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo -e "${RED}❌ Results directory not found: $RESULTS_DIR${NC}"
    exit 1
fi

# Find database file
DB_PATH="$RESULTS_DIR/simulation_data.db"
if [ ! -f "$DB_PATH" ]; then
    echo -e "${RED}❌ Database not found: $DB_PATH${NC}"
    exit 1
fi

echo -e "${BLUE}📂 Results directory: $RESULTS_DIR${NC}"
echo -e "${BLUE}💾 Database: $(basename "$DB_PATH")${NC}"

# Extract session name from directory if not provided
if [ -z "$SESSION_NAME" ]; then
    SESSION_NAME=$(basename "$RESULTS_DIR")
fi

# Build Rscript command
CMD="Rscript scripts/generate_fleet_dashboard.R --db-path \"$DB_PATH\" --session-name \"$SESSION_NAME\""

if [ -n "$EPISODE_ID" ]; then
    CMD="$CMD --episode-id $EPISODE_ID"
fi

echo -e "${BLUE}🚀 Running fleet analysis...${NC}"
echo ""

# Run the dashboard generation
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Success! Dashboard saved to:${NC}"
    echo -e "${GREEN}   $RESULTS_DIR/visualizations/${NC}"
    echo ""
    echo -e "${BLUE}📊 View the combined dashboard:${NC}"
    echo -e "${BLUE}   open $RESULTS_DIR/visualizations/fleet_dashboard_combined.png${NC}"
else
    echo -e "${RED}❌ Dashboard generation failed${NC}"
    exit 1
fi
