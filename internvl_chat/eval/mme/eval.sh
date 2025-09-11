#!/bin/bash

export DEBUG_MODE=1

### Testing the ordinary jpeg compression ###
# language_model='InternVL2_5-1B'
# method='exponential'
# prefilter='0'
# prescale='50.0'
# sigma_min='0.5'
# sigma_max='10.0'
# ksize='11'
# # for compress_ratio in 0 10 20 40 60 80; do
# for compress_ratio in 90; do
#      echo "Running with ratio=$compress_ratio"
#      python eval.py \
#             --compress-ratio $compress_ratio \
#             --dynamic

#     python calculation.py \
#         --results_dir $language_model-compress_$compress_ratio-method_$method-prefilter_$prefilter-prescale_$prescale-sigmamin_$sigma_min-sigmamax_$sigma_max-ksize_$ksize
# done

### Testing the prefiltering before ordinary jpeg compression ###
language_model='InternVL2_5-1B'
method='exponential'
prefilter='1'
prescale='80.0'
sigma_min='0.5'
sigma_max='3.0'
ksize='11'
for compress_ratio in 5 10 20 40 60 80 90; do
     echo "Running with ratio=$compress_ratio"
     python eval.py \
            --compress-ratio $compress_ratio \
            --dynamic \
            --prefilter \
            --prescale $prescale \
            --sigma_min $sigma_min \
            --sigma_max $sigma_max \
            --ksize $ksize

    python calculation.py \
        --results_dir $language_model-compress_$compress_ratio-method_$method-prefilter_$prefilter-prescale_$prescale-sigmamin_$sigma_min-sigmamax_$sigma_max-ksize_$ksize
done

### Testing the prefiltering with different sigma_max before fixed jpeg compression ###
# language_model='InternVL2_5-1B'
# method='exponential'
# prefilter='1'
# prescale='50.0'
# sigma_min='0.5'
# ksize='11'
# compress_ratio='70'
# for sigma_max in 0.5 0.75 1.0 1.5 2.0 5.0 10.0; do
#      echo "Running with sigma_max=$sigma_max"
#      python eval.py \
#             --compress-ratio $compress_ratio \
#             --dynamic \
#             --prefilter \
#             --prescale $prescale \
#             --sigma_min $sigma_min \
#             --sigma_max $sigma_max \
#             --ksize $ksize

#     python calculation.py \
#         --results_dir $language_model-compress_$compress_ratio-method_$method-prefilter_$prefilter-prescale_$prescale-sigmamin_$sigma_min-sigmamax_$sigma_max-ksize_$ksize
# done

# ### Testing the prefiltering with different prescale before fixed jpeg compression ###
# language_model='InternVL2_5-1B'
# method='exponential'
# prefilter='1'
# sigma_min='0.5'
# sigma_max='3.0'
# ksize='11'
# compress_ratio='70'
# for prescale in 1.0 10.0 20.0 50.0 80.0 100.0; do
#      echo "Running with prescale=$prescale"
#      python eval.py \
#             --compress-ratio $compress_ratio \
#             --dynamic \
#             --prefilter \
#             --prescale $prescale \
#             --sigma_min $sigma_min \
#             --sigma_max $sigma_max \
#             --ksize $ksize

#     python calculation.py \
#         --results_dir $language_model-compress_$compress_ratio-method_$method-prefilter_$prefilter-prescale_$prescale-sigmamin_$sigma_min-sigmamax_$sigma_max-ksize_$ksize
# done
