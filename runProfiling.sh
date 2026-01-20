#!/bin/bash
echo "Going to start profiling data collection! Output log file: ./cuda-profiling/profiling.log"
rm -f ./cuda-profiling/profiling.log
rm -f ./cuda-profiling/gpuData.csv
rm -f ./cuda-profiling/ncu-rep-results/*.ncu-rep
python ./cuda-profiling/gatherData.py --timeout 20 | tee ./cuda-profiling/profiling.log
echo "Profiling data collection complete!"