#!/bin/bash
source venv/bin/activate 
cat ${1} | parallel --gnu --lb -j 4 --tmpdir tmp/  eval {} --device {= '$_ = $job->slot() - 1' =} 
