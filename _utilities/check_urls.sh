#!/bin/bash

# Output file
output_file="url_check_results.txt"

# Initialize counters
total=0
working=0
not_found=0
other=0

# Create output file with header
echo "URL Status Check Results" > $output_file
echo "=========================" >> $output_file
echo "" >> $output_file

# Read each URL from the file
while read url; do
  # Skip empty lines
  if [ -z "$url" ]; then
    continue
  fi
  
  # Increment total counter
  ((total++))
  
  # Print progress
  echo "Checking $total: $url"
  
  # Use curl to check the URL status
  status_code=$(curl -s -o /dev/null -w "%{http_code}" "$url")
  
  # Check status code
  if [ "$status_code" -eq 200 ]; then
    echo "✓ WORKING: $url" >> $output_file
    ((working++))
  elif [ "$status_code" -eq 404 ]; then
    echo "✗ NOT FOUND (404): $url" >> $output_file
    ((not_found++))
  else
    echo "? OTHER STATUS ($status_code): $url" >> $output_file
    ((other++))
  fi
  
  # Small delay to avoid overwhelming the server
  sleep 0.1
  
done < old-nki-apis.txt

# Write summary
echo "" >> $output_file
echo "" >> $output_file
echo "Summary" >> $output_file
echo "=======" >> $output_file
echo "Total URLs checked: $total" >> $output_file
echo "Working URLs: $working" >> $output_file
echo "Not found (404) URLs: $not_found" >> $output_file
echo "Other status URLs: $other" >> $output_file

echo "URL check completed. Results saved to $output_file"
