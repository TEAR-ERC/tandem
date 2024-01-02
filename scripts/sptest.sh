# !/bin/bash
var=$(python /home/jyun/Tandem/check_exist.py /export/dump/jyun/perturb_stress/pert8_vs340/outputs/checkpoint/step1516954)
echo $var
if [ [ expr $var + 0 ] > 0 ]; then
    echo "here"
    exit $var
fi
echo "safety check passed"

