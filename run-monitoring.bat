@echo off
cd /d C:\Users\Skynet\Desktop\milk-mlops

echo ===== Monitoring started %date% %time% ===== >> reports\monitoring_job.log

docker compose --profile monitoring run --rm milk-monitoring >> reports\monitoring_job.log 2>&1

echo ===== Monitoring finished %date% %time% ===== >> reports\monitoring_job.log