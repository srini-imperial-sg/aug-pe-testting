mlm_prob=0.5
var_type="danielml_rephrase_tone"
feat_ext="sentence-t5-base"
length=64
temperature=1.4
num_seed_samples=554
# num_seed_samples=5
lookahead_degree=0
k=6 # number of variations
L=$((k+1))
init_L=${L} 
num_samples=$((L*num_seed_samples)) # 554 * 7 = 3878
echo generating $num_samples samples
epochs=10
word_var_scale=0
select_syn_mode=rank
# model_type=gpt2
model_type=meta-llama/Llama-3.2-1B-Instruct
noise=0
args=""
cls_batch_size=32
api="HFGPT"
feature_extractor_batch_size=512
if [ "$model_type" = "gpt2-large" ]; then
    batch_size=64
elif [ "$model_type" = "gpt2-medium" ]; then
    batch_size=128
elif [ "$model_type" = "gpt2" ]; then
    batch_size=512
else
    batch_size=256
fi

### load datacheckpoint 
data_checkpoint_args=""
for  (( iter=0; iter<=epochs; iter++ ))
do
train_file=${result_folder}/${iter}/samples.csv
if [ -e "$train_file" ]; then
    echo "$train_file does exist."
    # load from  data checkpoint
    data_checkpoint_args="--data_checkpoint_step ${iter} --data_checkpoint_path ${result_folder}/${iter}/samples.csv"
else
    echo "$train_file does not exist."
fi
done
echo load data from ${data_checkpoint_args} ${args}
# threshold eps 0.5 break_noise 20.50000000000388 eps 0.500195
# threshold eps 1 break_noise 10.960000000005778 eps 1.000801
# threshold eps 2 break_noise 5.900000000006784 eps 2.003615
# threshold eps 4 break_noise 3.2300000000073155 eps 4.002833
# "3.23" "5.9" "10.96" "20.5"
for noise in "5.9"; do 
    echo "Noise level ${noise}."
    result_folder="result/Daniel-ML/${model_type}_${feat_ext//\//_}/${num_samples}_n${noise}_L${L}_initL${init_L}_var${lookahead_degree}_${var_type}_${select_syn_mode}_len${length}var${word_var_scale}_t${temperature}"
    echo $result_folder
    mkdir -p $result_folder
    ### run PE
    python main.py ${args} ${data_checkpoint_args} \
    --dataset cls/Daniel-ML \
    --train_data_file ../../data/cls/Daniel-ML/original/train_original.jsonl \
    --api ${api} \
    --noise ${noise} \
    --model_type ${model_type} \
    --do_sample  \
    --length ${length} \
    --random_sampling_batch_size ${batch_size} \
    --variation_batch_size ${batch_size} \
    --temperature ${temperature} \
    --select_syn_mode ${select_syn_mode} \
    --num_samples_schedule ${num_samples} \
    --combine_divide_L ${L} \
    --init_combine_divide_L ${init_L} \
    --variation_degree_schedule ${mlm_prob} \
    --lookahead_degree ${lookahead_degree} \
    --epochs ${epochs} \
    --feature_extractor ${feat_ext} \
    --feature_extractor_batch_size ${feature_extractor_batch_size} \
    --mlm_probability ${mlm_prob} \
    --variation_type ${var_type} \
    --result_folder ${result_folder} \
    --log_online \
    --apply_template \
    --train_data_embeddings_file result/embeddings/${feat_ext//\//_}/cls_Daniel-ML_train_all.embeddings.npz #> $result_folder/output.log 2>&1
done