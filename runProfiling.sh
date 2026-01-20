#!/bin/bash
echo "Going to start profiling data collection! Output log file: ./cuda-profiling/profiling.log"
python ./cuda-profiling/gatherData.py | tee ./cuda-profiling/profiling.log
echo "Profiling data collection complete!"