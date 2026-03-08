#!/usr/bin/env bash
# Provision SigNoz dashboards via API
# Usage: ./scripts/provision_dashboards.sh [SIGNOZ_URL]
#
# Default: http://localhost:8080 (dev) or adjust for production

set -euo pipefail

SIGNOZ_URL="${1:-http://localhost:8080}"
DASHBOARDS_DIR="$(dirname "$0")/../signoz/dashboards"

echo "=== Provisioning SigNoz Dashboards ==="
echo "SigNoz URL: ${SIGNOZ_URL}"
echo ""

# Wait for SigNoz to be healthy
echo "Waiting for SigNoz to be ready..."
for i in $(seq 1 30); do
    if curl -sf "${SIGNOZ_URL}/api/v1/health" > /dev/null 2>&1; then
        echo "SigNoz is ready!"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: SigNoz not reachable at ${SIGNOZ_URL} after 30 attempts"
        exit 1
    fi
    sleep 2
done

echo ""

# Import each dashboard JSON
for dashboard_file in "${DASHBOARDS_DIR}"/*.json; do
    [ -f "$dashboard_file" ] || continue

    name=$(basename "$dashboard_file" .json)
    title=$(python3 -c "import json; print(json.load(open('${dashboard_file}'))['title'])" 2>/dev/null || echo "$name")

    echo "Importing: ${title} (${name}.json)"

    # Wrap the dashboard data in the expected API format
    payload=$(python3 -c "
import json, sys
with open('${dashboard_file}') as f:
    data = json.load(f)
# The API expects the dashboard data directly
print(json.dumps(data))
")

    response=$(curl -sf -X POST "${SIGNOZ_URL}/api/v1/dashboards" \
        -H "Content-Type: application/json" \
        -d "${payload}" 2>&1) && {
        echo "  OK"
    } || {
        echo "  WARN: Could not import (may already exist or auth required)"
        echo "  Response: ${response}"
    }
done

echo ""
echo "Done! Access dashboards at: ${SIGNOZ_URL}/dashboard"
echo ""
echo "If dashboards require authentication, you can import them manually:"
echo "  1. Open ${SIGNOZ_URL}/dashboard"
echo "  2. Click 'New Dashboard' → 'Import JSON'"
echo "  3. Paste the contents of the JSON files from signoz/dashboards/"
