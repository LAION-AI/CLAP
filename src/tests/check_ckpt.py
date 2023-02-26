import torch

def keys_in_state_dict(ckpt, device='cpu'):
    if device=="cpu":
        a = torch.load(ckpt, map_location=torch.device('cpu'))["state_dict"]
    else:
        a = torch.load(ckpt)["state_dict"]
    print("keys_in_state_dict", a.keys())


def check_ckpt_diff(ckpt_a, ckpt_b, key_include=None, key_exclude=None, device='cpu', verbose=True):
    if device=="cpu":
        a = torch.load(ckpt_a, map_location=torch.device('cpu'))["state_dict"]
        b = torch.load(ckpt_b, map_location=torch.device('cpu'))["state_dict"]
    else:
        a = torch.load(ckpt_a)["state_dict"]
        b = torch.load(ckpt_b)["state_dict"]
    a_sum = 0
    b_sum = 0
    difference_count = 0
    for k in a.keys():
        if key_include is not None and key_include not in k:
            continue
        if key_exclude is not None and key_exclude in k:
            continue
        if k in b.keys():
            a_sum += torch.sum(a[k])
            b_sum += torch.sum(b[k])
            if verbose:
                if torch.sum(a[k]) != torch.sum(b[k]):
                    print(f"key {k} is different")
                    difference_count += 1
    print("a_sum: ", a_sum)
    print("b_sum: ", b_sum)
    print("diff: ", a_sum - b_sum)
    if verbose:
        print("difference_count: ", difference_count)
    return bool(a_sum - b_sum)

# Transformer no freeze:
# check_ckpt_diff("/fsx/clap_logs/2022_09_11-19_37_08-model_PANN-14-lr_0.001-b_160-j_4-p_fp32/checkpoints/epoch_10.pt", "/fsx/clap_logs/2022_09_11-19_37_08-model_PANN-14-lr_0.001-b_160-j_4-p_fp32/checkpoints/epoch_100.pt", "text_branch.resblocks")

check_ckpt_diff("/fsx/clap_logs/2022_09_29-23_42_40-model_PANN-14-lr_0.001-b_160-j_4-p_fp32/checkpoints/epoch_1.pt",
                "/fsx/clap_logs/2022_09_29-23_42_40-model_PANN-14-lr_0.001-b_160-j_4-p_fp32/checkpoints/epoch_2.pt",
                "text_branch.resblocks")

# key module.text_branch.resblocks.0.attn.in_proj_weight is different
# key module.text_branch.resblocks.0.attn.in_proj_bias is different
# key module.text_branch.resblocks.0.attn.out_proj.weight is different
# key module.text_branch.resblocks.0.attn.out_proj.bias is different
# key module.text_branch.resblocks.0.ln_1.weight is different
# key module.text_branch.resblocks.0.ln_1.bias is different
# key module.text_branch.resblocks.0.mlp.c_fc.weight is different
# key module.text_branch.resblocks.0.mlp.c_fc.bias is different
# key module.text_branch.resblocks.0.mlp.c_proj.weight is different
# key module.text_branch.resblocks.0.mlp.c_proj.bias is different
# key module.text_branch.resblocks.0.ln_2.weight is different
# key module.text_branch.resblocks.0.ln_2.bias is different
# key module.text_branch.resblocks.1.attn.in_proj_weight is different
# key module.text_branch.resblocks.1.attn.in_proj_bias is different
# key module.text_branch.resblocks.1.attn.out_proj.weight is different
# key module.text_branch.resblocks.1.attn.out_proj.bias is different
# key module.text_branch.resblocks.1.ln_1.weight is different
# key module.text_branch.resblocks.1.ln_1.bias is different
# key module.text_branch.resblocks.1.mlp.c_fc.weight is different
# key module.text_branch.resblocks.1.mlp.c_fc.bias is different
# key module.text_branch.resblocks.1.mlp.c_proj.weight is different
# key module.text_branch.resblocks.1.mlp.c_proj.bias is different
# key module.text_branch.resblocks.1.ln_2.weight is different
# key module.text_branch.resblocks.1.ln_2.bias is different
# key module.text_branch.resblocks.2.attn.in_proj_weight is different
# key module.text_branch.resblocks.2.attn.in_proj_bias is different
# key module.text_branch.resblocks.2.attn.out_proj.weight is different
# key module.text_branch.resblocks.2.attn.out_proj.bias is different
# key module.text_branch.resblocks.2.ln_1.weight is different
# key module.text_branch.resblocks.2.ln_1.bias is different
# key module.text_branch.resblocks.2.mlp.c_fc.weight is different
# key module.text_branch.resblocks.2.mlp.c_fc.bias is different
# key module.text_branch.resblocks.2.mlp.c_proj.weight is different
# key module.text_branch.resblocks.2.mlp.c_proj.bias is different
# key module.text_branch.resblocks.2.ln_2.weight is different
# key module.text_branch.resblocks.2.ln_2.bias is different
# key module.text_branch.resblocks.3.attn.in_proj_weight is different
# key module.text_branch.resblocks.3.attn.in_proj_bias is different
# key module.text_branch.resblocks.3.attn.out_proj.weight is different
# key module.text_branch.resblocks.3.attn.out_proj.bias is different
# key module.text_branch.resblocks.3.ln_1.weight is different
# key module.text_branch.resblocks.3.ln_1.bias is different
# key module.text_branch.resblocks.3.mlp.c_fc.weight is different
# key module.text_branch.resblocks.3.mlp.c_fc.bias is different
# key module.text_branch.resblocks.3.mlp.c_proj.weight is different
# key module.text_branch.resblocks.3.mlp.c_proj.bias is different
# key module.text_branch.resblocks.3.ln_2.weight is different
# key module.text_branch.resblocks.3.ln_2.bias is different
# key module.text_branch.resblocks.4.attn.in_proj_weight is different
# key module.text_branch.resblocks.4.attn.in_proj_bias is different
# key module.text_branch.resblocks.4.attn.out_proj.weight is different
# key module.text_branch.resblocks.4.attn.out_proj.bias is different
# key module.text_branch.resblocks.4.ln_1.weight is different
# key module.text_branch.resblocks.4.ln_1.bias is different
# key module.text_branch.resblocks.4.mlp.c_fc.weight is different
# key module.text_branch.resblocks.4.mlp.c_fc.bias is different
# key module.text_branch.resblocks.4.mlp.c_proj.weight is different
# key module.text_branch.resblocks.4.mlp.c_proj.bias is different
# key module.text_branch.resblocks.4.ln_2.weight is different
# key module.text_branch.resblocks.4.ln_2.bias is different
# key module.text_branch.resblocks.5.attn.in_proj_weight is different
# key module.text_branch.resblocks.5.attn.in_proj_bias is different
# key module.text_branch.resblocks.5.attn.out_proj.weight is different
# key module.text_branch.resblocks.5.attn.out_proj.bias is different
# key module.text_branch.resblocks.5.ln_1.weight is different
# key module.text_branch.resblocks.5.ln_1.bias is different
# key module.text_branch.resblocks.5.mlp.c_fc.weight is different
# key module.text_branch.resblocks.5.mlp.c_fc.bias is different
# key module.text_branch.resblocks.5.mlp.c_proj.weight is different
# key module.text_branch.resblocks.5.mlp.c_proj.bias is different
# key module.text_branch.resblocks.5.ln_2.weight is different
# key module.text_branch.resblocks.5.ln_2.bias is different
# key module.text_branch.resblocks.6.attn.in_proj_weight is different
# key module.text_branch.resblocks.6.attn.in_proj_bias is different
# key module.text_branch.resblocks.6.attn.out_proj.weight is different
# key module.text_branch.resblocks.6.attn.out_proj.bias is different
# key module.text_branch.resblocks.6.ln_1.weight is different
# key module.text_branch.resblocks.6.ln_1.bias is different
# key module.text_branch.resblocks.6.mlp.c_fc.weight is different
# key module.text_branch.resblocks.6.mlp.c_fc.bias is different
# key module.text_branch.resblocks.6.mlp.c_proj.weight is different
# key module.text_branch.resblocks.6.mlp.c_proj.bias is different
# key module.text_branch.resblocks.6.ln_2.weight is different
# key module.text_branch.resblocks.6.ln_2.bias is different
# key module.text_branch.resblocks.7.attn.in_proj_weight is different
# key module.text_branch.resblocks.7.attn.in_proj_bias is different
# key module.text_branch.resblocks.7.attn.out_proj.weight is different
# key module.text_branch.resblocks.7.attn.out_proj.bias is different
# key module.text_branch.resblocks.7.ln_1.weight is different
# key module.text_branch.resblocks.7.ln_1.bias is different
# key module.text_branch.resblocks.7.mlp.c_fc.weight is different
# key module.text_branch.resblocks.7.mlp.c_fc.bias is different
# key module.text_branch.resblocks.7.mlp.c_proj.weight is different
# key module.text_branch.resblocks.7.mlp.c_proj.bias is different
# key module.text_branch.resblocks.7.ln_2.weight is different
# key module.text_branch.resblocks.7.ln_2.bias is different
# key module.text_branch.resblocks.8.attn.in_proj_weight is different
# key module.text_branch.resblocks.8.attn.in_proj_bias is different
# key module.text_branch.resblocks.8.attn.out_proj.weight is different
# key module.text_branch.resblocks.8.attn.out_proj.bias is different
# key module.text_branch.resblocks.8.ln_1.weight is different
# key module.text_branch.resblocks.8.ln_1.bias is different
# key module.text_branch.resblocks.8.mlp.c_fc.weight is different
# key module.text_branch.resblocks.8.mlp.c_fc.bias is different
# key module.text_branch.resblocks.8.mlp.c_proj.weight is different
# key module.text_branch.resblocks.8.mlp.c_proj.bias is different
# key module.text_branch.resblocks.8.ln_2.weight is different
# key module.text_branch.resblocks.8.ln_2.bias is different
# key module.text_branch.resblocks.9.attn.in_proj_weight is different
# key module.text_branch.resblocks.9.attn.in_proj_bias is different
# key module.text_branch.resblocks.9.attn.out_proj.weight is different
# key module.text_branch.resblocks.9.attn.out_proj.bias is different
# key module.text_branch.resblocks.9.ln_1.weight is different
# key module.text_branch.resblocks.9.ln_1.bias is different
# key module.text_branch.resblocks.9.mlp.c_fc.weight is different
# key module.text_branch.resblocks.9.mlp.c_fc.bias is different
# key module.text_branch.resblocks.9.mlp.c_proj.weight is different
# key module.text_branch.resblocks.9.mlp.c_proj.bias is different
# key module.text_branch.resblocks.9.ln_2.weight is different
# key module.text_branch.resblocks.9.ln_2.bias is different
# key module.text_branch.resblocks.10.attn.in_proj_weight is different
# key module.text_branch.resblocks.10.attn.in_proj_bias is different
# key module.text_branch.resblocks.10.attn.out_proj.weight is different
# key module.text_branch.resblocks.10.attn.out_proj.bias is different
# key module.text_branch.resblocks.10.ln_1.weight is different
# key module.text_branch.resblocks.10.ln_1.bias is different
# key module.text_branch.resblocks.10.mlp.c_fc.weight is different
# key module.text_branch.resblocks.10.mlp.c_fc.bias is different
# key module.text_branch.resblocks.10.mlp.c_proj.weight is different
# key module.text_branch.resblocks.10.mlp.c_proj.bias is different
# key module.text_branch.resblocks.10.ln_2.weight is different
# key module.text_branch.resblocks.10.ln_2.bias is different
# key module.text_branch.resblocks.11.attn.in_proj_weight is different
# key module.text_branch.resblocks.11.attn.in_proj_bias is different
# key module.text_branch.resblocks.11.attn.out_proj.weight is different
# key module.text_branch.resblocks.11.attn.out_proj.bias is different
# key module.text_branch.resblocks.11.ln_1.weight is different
# key module.text_branch.resblocks.11.ln_1.bias is different
# key module.text_branch.resblocks.11.mlp.c_fc.weight is different
# key module.text_branch.resblocks.11.mlp.c_fc.bias is different
# key module.text_branch.resblocks.11.mlp.c_proj.weight is different
# key module.text_branch.resblocks.11.mlp.c_proj.bias is different
# key module.text_branch.resblocks.11.ln_2.weight is different
# key module.text_branch.resblocks.11.ln_2.bias is different
# a_sum:  tensor(12113.6445)
# b_sum:  tensor(9883.4424)
# diff:  tensor(2230.2021)
# True


# Transformer freeze:
# check_ckpt_diff("/fsx/clap_logs/2022_09_16-18_55_10-model_PANN-14-lr_0.001-b_160-j_4-p_fp32/checkpoints/epoch_10.pt", "/fsx/clap_logs/2022_09_16-18_55_10-model_PANN-14-lr_0.001-b_160-j_4-p_fp32/checkpoints/epoch_100.pt", "text_branch.resblocks")

# key module.text_branch.resblocks.0.attn.in_proj_weight is different
# key module.text_branch.resblocks.0.attn.in_proj_bias is different
# key module.text_branch.resblocks.0.attn.out_proj.weight is different
# key module.text_branch.resblocks.0.attn.out_proj.bias is different
# key module.text_branch.resblocks.0.ln_1.weight is different
# key module.text_branch.resblocks.0.ln_1.bias is different
# key module.text_branch.resblocks.0.mlp.c_fc.weight is different
# key module.text_branch.resblocks.0.mlp.c_fc.bias is different
# key module.text_branch.resblocks.0.mlp.c_proj.weight is different
# key module.text_branch.resblocks.0.mlp.c_proj.bias is different
# key module.text_branch.resblocks.0.ln_2.weight is different
# key module.text_branch.resblocks.0.ln_2.bias is different
# key module.text_branch.resblocks.1.attn.in_proj_weight is different
# key module.text_branch.resblocks.1.attn.in_proj_bias is different
# key module.text_branch.resblocks.1.attn.out_proj.weight is different
# key module.text_branch.resblocks.1.attn.out_proj.bias is different
# key module.text_branch.resblocks.1.ln_1.weight is different
# key module.text_branch.resblocks.1.ln_1.bias is different
# key module.text_branch.resblocks.1.mlp.c_fc.weight is different
# key module.text_branch.resblocks.1.mlp.c_fc.bias is different
# key module.text_branch.resblocks.1.mlp.c_proj.weight is different
# key module.text_branch.resblocks.1.mlp.c_proj.bias is different
# key module.text_branch.resblocks.1.ln_2.weight is different
# key module.text_branch.resblocks.1.ln_2.bias is different
# key module.text_branch.resblocks.2.attn.in_proj_weight is different
# key module.text_branch.resblocks.2.attn.in_proj_bias is different
# key module.text_branch.resblocks.2.attn.out_proj.weight is different
# key module.text_branch.resblocks.2.attn.out_proj.bias is different
# key module.text_branch.resblocks.2.ln_1.weight is different
# key module.text_branch.resblocks.2.ln_1.bias is different
# key module.text_branch.resblocks.2.mlp.c_fc.weight is different
# key module.text_branch.resblocks.2.mlp.c_fc.bias is different
# key module.text_branch.resblocks.2.mlp.c_proj.weight is different
# key module.text_branch.resblocks.2.mlp.c_proj.bias is different
# key module.text_branch.resblocks.2.ln_2.weight is different
# key module.text_branch.resblocks.2.ln_2.bias is different
# key module.text_branch.resblocks.3.attn.in_proj_weight is different
# key module.text_branch.resblocks.3.attn.in_proj_bias is different
# key module.text_branch.resblocks.3.attn.out_proj.weight is different
# key module.text_branch.resblocks.3.attn.out_proj.bias is different
# key module.text_branch.resblocks.3.ln_1.weight is different
# key module.text_branch.resblocks.3.ln_1.bias is different
# key module.text_branch.resblocks.3.mlp.c_fc.weight is different
# key module.text_branch.resblocks.3.mlp.c_fc.bias is different
# key module.text_branch.resblocks.3.mlp.c_proj.weight is different
# key module.text_branch.resblocks.3.mlp.c_proj.bias is different
# key module.text_branch.resblocks.3.ln_2.weight is different
# key module.text_branch.resblocks.3.ln_2.bias is different
# key module.text_branch.resblocks.4.attn.in_proj_weight is different
# key module.text_branch.resblocks.4.attn.in_proj_bias is different
# key module.text_branch.resblocks.4.attn.out_proj.weight is different
# key module.text_branch.resblocks.4.attn.out_proj.bias is different
# key module.text_branch.resblocks.4.ln_1.weight is different
# key module.text_branch.resblocks.4.ln_1.bias is different
# key module.text_branch.resblocks.4.mlp.c_fc.weight is different
# key module.text_branch.resblocks.4.mlp.c_fc.bias is different
# key module.text_branch.resblocks.4.mlp.c_proj.weight is different
# key module.text_branch.resblocks.4.mlp.c_proj.bias is different
# key module.text_branch.resblocks.4.ln_2.weight is different
# key module.text_branch.resblocks.4.ln_2.bias is different
# key module.text_branch.resblocks.5.attn.in_proj_weight is different
# key module.text_branch.resblocks.5.attn.in_proj_bias is different
# key module.text_branch.resblocks.5.attn.out_proj.weight is different
# key module.text_branch.resblocks.5.attn.out_proj.bias is different
# key module.text_branch.resblocks.5.ln_1.weight is different
# key module.text_branch.resblocks.5.ln_1.bias is different
# key module.text_branch.resblocks.5.mlp.c_fc.weight is different
# key module.text_branch.resblocks.5.mlp.c_fc.bias is different
# key module.text_branch.resblocks.5.mlp.c_proj.weight is different
# key module.text_branch.resblocks.5.mlp.c_proj.bias is different
# key module.text_branch.resblocks.5.ln_2.weight is different
# key module.text_branch.resblocks.5.ln_2.bias is different
# key module.text_branch.resblocks.6.attn.in_proj_weight is different
# key module.text_branch.resblocks.6.attn.in_proj_bias is different
# key module.text_branch.resblocks.6.attn.out_proj.weight is different
# key module.text_branch.resblocks.6.attn.out_proj.bias is different
# key module.text_branch.resblocks.6.ln_1.weight is different
# key module.text_branch.resblocks.6.ln_1.bias is different
# key module.text_branch.resblocks.6.mlp.c_fc.weight is different
# key module.text_branch.resblocks.6.mlp.c_fc.bias is different
# key module.text_branch.resblocks.6.mlp.c_proj.weight is different
# key module.text_branch.resblocks.6.mlp.c_proj.bias is different
# key module.text_branch.resblocks.6.ln_2.weight is different
# key module.text_branch.resblocks.6.ln_2.bias is different
# key module.text_branch.resblocks.7.attn.in_proj_weight is different
# key module.text_branch.resblocks.7.attn.in_proj_bias is different
# key module.text_branch.resblocks.7.attn.out_proj.weight is different
# key module.text_branch.resblocks.7.attn.out_proj.bias is different
# key module.text_branch.resblocks.7.ln_1.weight is different
# key module.text_branch.resblocks.7.ln_1.bias is different
# key module.text_branch.resblocks.7.mlp.c_fc.weight is different
# key module.text_branch.resblocks.7.mlp.c_fc.bias is different
# key module.text_branch.resblocks.7.mlp.c_proj.weight is different
# key module.text_branch.resblocks.7.mlp.c_proj.bias is different
# key module.text_branch.resblocks.7.ln_2.weight is different
# key module.text_branch.resblocks.7.ln_2.bias is different
# key module.text_branch.resblocks.8.attn.in_proj_weight is different
# key module.text_branch.resblocks.8.attn.in_proj_bias is different
# key module.text_branch.resblocks.8.attn.out_proj.weight is different
# key module.text_branch.resblocks.8.attn.out_proj.bias is different
# key module.text_branch.resblocks.8.ln_1.weight is different
# key module.text_branch.resblocks.8.ln_1.bias is different
# key module.text_branch.resblocks.8.mlp.c_fc.weight is different
# key module.text_branch.resblocks.8.mlp.c_fc.bias is different
# key module.text_branch.resblocks.8.mlp.c_proj.weight is different
# key module.text_branch.resblocks.8.mlp.c_proj.bias is different
# key module.text_branch.resblocks.8.ln_2.weight is different
# key module.text_branch.resblocks.8.ln_2.bias is different
# key module.text_branch.resblocks.9.attn.in_proj_weight is different
# key module.text_branch.resblocks.9.attn.in_proj_bias is different
# key module.text_branch.resblocks.9.attn.out_proj.weight is different
# key module.text_branch.resblocks.9.attn.out_proj.bias is different
# key module.text_branch.resblocks.9.ln_1.weight is different
# key module.text_branch.resblocks.9.ln_1.bias is different
# key module.text_branch.resblocks.9.mlp.c_fc.weight is different
# key module.text_branch.resblocks.9.mlp.c_fc.bias is different
# key module.text_branch.resblocks.9.mlp.c_proj.weight is different
# key module.text_branch.resblocks.9.mlp.c_proj.bias is different
# key module.text_branch.resblocks.9.ln_2.weight is different
# key module.text_branch.resblocks.9.ln_2.bias is different
# key module.text_branch.resblocks.10.attn.in_proj_weight is different
# key module.text_branch.resblocks.10.attn.in_proj_bias is different
# key module.text_branch.resblocks.10.attn.out_proj.weight is different
# key module.text_branch.resblocks.10.attn.out_proj.bias is different
# key module.text_branch.resblocks.10.ln_1.weight is different
# key module.text_branch.resblocks.10.ln_1.bias is different
# key module.text_branch.resblocks.10.mlp.c_fc.weight is different
# key module.text_branch.resblocks.10.mlp.c_fc.bias is different
# key module.text_branch.resblocks.10.mlp.c_proj.weight is different
# key module.text_branch.resblocks.10.mlp.c_proj.bias is different
# key module.text_branch.resblocks.10.ln_2.weight is different
# key module.text_branch.resblocks.10.ln_2.bias is different
# key module.text_branch.resblocks.11.attn.in_proj_weight is different
# key module.text_branch.resblocks.11.attn.in_proj_bias is different
# key module.text_branch.resblocks.11.attn.out_proj.weight is different
# key module.text_branch.resblocks.11.attn.out_proj.bias is different
# key module.text_branch.resblocks.11.ln_1.weight is different
# key module.text_branch.resblocks.11.ln_1.bias is different
# key module.text_branch.resblocks.11.mlp.c_fc.weight is different
# key module.text_branch.resblocks.11.mlp.c_fc.bias is different
# key module.text_branch.resblocks.11.mlp.c_proj.weight is different
# key module.text_branch.resblocks.11.mlp.c_proj.bias is different
# key module.text_branch.resblocks.11.ln_2.weight is different
# key module.text_branch.resblocks.11.ln_2.bias is different
# a_sum:  tensor(12133.6348)
# b_sum:  tensor(10423.9521)
# diff:  tensor(1709.6826)
# True


# bert no freeze:
# check_ckpt_diff("/fsx/clap_logs/2022_09_14-02_33_11-model_PANN-14-lr_0.0001-b_160-j_4-p_fp32/checkpoints/epoch_10.pt", "/fsx/clap_logs/2022_09_14-02_33_11-model_PANN-14-lr_0.0001-b_160-j_4-p_fp32/checkpoints/epoch_100.pt", "text_branch.encoder")

# key module.text_branch.encoder.layer.0.attention.self.query.weight is different
# key module.text_branch.encoder.layer.0.attention.self.query.bias is different
# key module.text_branch.encoder.layer.0.attention.self.key.weight is different
# key module.text_branch.encoder.layer.0.attention.self.key.bias is different
# key module.text_branch.encoder.layer.0.attention.self.value.weight is different
# key module.text_branch.encoder.layer.0.attention.self.value.bias is different
# key module.text_branch.encoder.layer.0.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.0.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.0.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.0.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.0.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.0.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.0.output.dense.weight is different
# key module.text_branch.encoder.layer.0.output.dense.bias is different
# key module.text_branch.encoder.layer.0.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.0.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.1.attention.self.query.weight is different
# key module.text_branch.encoder.layer.1.attention.self.query.bias is different
# key module.text_branch.encoder.layer.1.attention.self.key.weight is different
# key module.text_branch.encoder.layer.1.attention.self.key.bias is different
# key module.text_branch.encoder.layer.1.attention.self.value.weight is different
# key module.text_branch.encoder.layer.1.attention.self.value.bias is different
# key module.text_branch.encoder.layer.1.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.1.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.1.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.1.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.1.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.1.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.1.output.dense.weight is different
# key module.text_branch.encoder.layer.1.output.dense.bias is different
# key module.text_branch.encoder.layer.1.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.1.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.2.attention.self.query.weight is different
# key module.text_branch.encoder.layer.2.attention.self.query.bias is different
# key module.text_branch.encoder.layer.2.attention.self.key.weight is different
# key module.text_branch.encoder.layer.2.attention.self.key.bias is different
# key module.text_branch.encoder.layer.2.attention.self.value.weight is different
# key module.text_branch.encoder.layer.2.attention.self.value.bias is different
# key module.text_branch.encoder.layer.2.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.2.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.2.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.2.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.2.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.2.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.2.output.dense.weight is different
# key module.text_branch.encoder.layer.2.output.dense.bias is different
# key module.text_branch.encoder.layer.2.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.2.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.3.attention.self.query.weight is different
# key module.text_branch.encoder.layer.3.attention.self.query.bias is different
# key module.text_branch.encoder.layer.3.attention.self.key.weight is different
# key module.text_branch.encoder.layer.3.attention.self.key.bias is different
# key module.text_branch.encoder.layer.3.attention.self.value.weight is different
# key module.text_branch.encoder.layer.3.attention.self.value.bias is different
# key module.text_branch.encoder.layer.3.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.3.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.3.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.3.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.3.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.3.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.3.output.dense.weight is different
# key module.text_branch.encoder.layer.3.output.dense.bias is different
# key module.text_branch.encoder.layer.3.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.3.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.4.attention.self.query.weight is different
# key module.text_branch.encoder.layer.4.attention.self.query.bias is different
# key module.text_branch.encoder.layer.4.attention.self.key.weight is different
# key module.text_branch.encoder.layer.4.attention.self.key.bias is different
# key module.text_branch.encoder.layer.4.attention.self.value.weight is different
# key module.text_branch.encoder.layer.4.attention.self.value.bias is different
# key module.text_branch.encoder.layer.4.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.4.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.4.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.4.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.4.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.4.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.4.output.dense.weight is different
# key module.text_branch.encoder.layer.4.output.dense.bias is different
# key module.text_branch.encoder.layer.4.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.4.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.5.attention.self.query.weight is different
# key module.text_branch.encoder.layer.5.attention.self.query.bias is different
# key module.text_branch.encoder.layer.5.attention.self.key.weight is different
# key module.text_branch.encoder.layer.5.attention.self.key.bias is different
# key module.text_branch.encoder.layer.5.attention.self.value.weight is different
# key module.text_branch.encoder.layer.5.attention.self.value.bias is different
# key module.text_branch.encoder.layer.5.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.5.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.5.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.5.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.5.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.5.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.5.output.dense.weight is different
# key module.text_branch.encoder.layer.5.output.dense.bias is different
# key module.text_branch.encoder.layer.5.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.5.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.6.attention.self.query.weight is different
# key module.text_branch.encoder.layer.6.attention.self.query.bias is different
# key module.text_branch.encoder.layer.6.attention.self.key.weight is different
# key module.text_branch.encoder.layer.6.attention.self.key.bias is different
# key module.text_branch.encoder.layer.6.attention.self.value.weight is different
# key module.text_branch.encoder.layer.6.attention.self.value.bias is different
# key module.text_branch.encoder.layer.6.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.6.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.6.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.6.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.6.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.6.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.6.output.dense.weight is different
# key module.text_branch.encoder.layer.6.output.dense.bias is different
# key module.text_branch.encoder.layer.6.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.6.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.7.attention.self.query.weight is different
# key module.text_branch.encoder.layer.7.attention.self.query.bias is different
# key module.text_branch.encoder.layer.7.attention.self.key.weight is different
# key module.text_branch.encoder.layer.7.attention.self.key.bias is different
# key module.text_branch.encoder.layer.7.attention.self.value.weight is different
# key module.text_branch.encoder.layer.7.attention.self.value.bias is different
# key module.text_branch.encoder.layer.7.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.7.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.7.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.7.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.7.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.7.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.7.output.dense.weight is different
# key module.text_branch.encoder.layer.7.output.dense.bias is different
# key module.text_branch.encoder.layer.7.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.7.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.8.attention.self.query.weight is different
# key module.text_branch.encoder.layer.8.attention.self.query.bias is different
# key module.text_branch.encoder.layer.8.attention.self.key.weight is different
# key module.text_branch.encoder.layer.8.attention.self.key.bias is different
# key module.text_branch.encoder.layer.8.attention.self.value.weight is different
# key module.text_branch.encoder.layer.8.attention.self.value.bias is different
# key module.text_branch.encoder.layer.8.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.8.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.8.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.8.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.8.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.8.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.8.output.dense.weight is different
# key module.text_branch.encoder.layer.8.output.dense.bias is different
# key module.text_branch.encoder.layer.8.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.8.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.9.attention.self.query.weight is different
# key module.text_branch.encoder.layer.9.attention.self.query.bias is different
# key module.text_branch.encoder.layer.9.attention.self.key.weight is different
# key module.text_branch.encoder.layer.9.attention.self.key.bias is different
# key module.text_branch.encoder.layer.9.attention.self.value.weight is different
# key module.text_branch.encoder.layer.9.attention.self.value.bias is different
# key module.text_branch.encoder.layer.9.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.9.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.9.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.9.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.9.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.9.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.9.output.dense.weight is different
# key module.text_branch.encoder.layer.9.output.dense.bias is different
# key module.text_branch.encoder.layer.9.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.9.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.10.attention.self.query.weight is different
# key module.text_branch.encoder.layer.10.attention.self.query.bias is different
# key module.text_branch.encoder.layer.10.attention.self.key.weight is different
# key module.text_branch.encoder.layer.10.attention.self.key.bias is different
# key module.text_branch.encoder.layer.10.attention.self.value.weight is different
# key module.text_branch.encoder.layer.10.attention.self.value.bias is different
# key module.text_branch.encoder.layer.10.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.10.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.10.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.10.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.10.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.10.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.10.output.dense.weight is different
# key module.text_branch.encoder.layer.10.output.dense.bias is different
# key module.text_branch.encoder.layer.10.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.10.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.11.attention.self.query.weight is different
# key module.text_branch.encoder.layer.11.attention.self.query.bias is different
# key module.text_branch.encoder.layer.11.attention.self.key.weight is different
# key module.text_branch.encoder.layer.11.attention.self.key.bias is different
# key module.text_branch.encoder.layer.11.attention.self.value.weight is different
# key module.text_branch.encoder.layer.11.attention.self.value.bias is different
# key module.text_branch.encoder.layer.11.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.11.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.11.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.11.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.11.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.11.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.11.output.dense.weight is different
# key module.text_branch.encoder.layer.11.output.dense.bias is different
# key module.text_branch.encoder.layer.11.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.11.output.LayerNorm.bias is different
# a_sum:  tensor(15185.1230)
# b_sum:  tensor(15576.5596)
# diff:  tensor(-391.4365)
# True


# bert freeze:
# check_ckpt_diff("/fsx/clap_logs/2022_09_13-01_25_15-model_PANN-14-lr_0.0001-b_160-j_4-p_fp32/checkpoints/epoch_10.pt", "/fsx/clap_logs/2022_09_13-01_25_15-model_PANN-14-lr_0.0001-b_160-j_4-p_fp32/checkpoints/epoch_100.pt", "text_branch.encoder")

# key module.text_branch.encoder.layer.0.attention.self.query.weight is different
# key module.text_branch.encoder.layer.0.attention.self.query.bias is different
# key module.text_branch.encoder.layer.0.attention.self.key.weight is different
# key module.text_branch.encoder.layer.0.attention.self.key.bias is different
# key module.text_branch.encoder.layer.0.attention.self.value.weight is different
# key module.text_branch.encoder.layer.0.attention.self.value.bias is different
# key module.text_branch.encoder.layer.0.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.0.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.0.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.0.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.0.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.0.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.0.output.dense.weight is different
# key module.text_branch.encoder.layer.0.output.dense.bias is different
# key module.text_branch.encoder.layer.0.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.0.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.1.attention.self.query.weight is different
# key module.text_branch.encoder.layer.1.attention.self.query.bias is different
# key module.text_branch.encoder.layer.1.attention.self.key.weight is different
# key module.text_branch.encoder.layer.1.attention.self.key.bias is different
# key module.text_branch.encoder.layer.1.attention.self.value.weight is different
# key module.text_branch.encoder.layer.1.attention.self.value.bias is different
# key module.text_branch.encoder.layer.1.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.1.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.1.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.1.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.1.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.1.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.1.output.dense.weight is different
# key module.text_branch.encoder.layer.1.output.dense.bias is different
# key module.text_branch.encoder.layer.1.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.1.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.2.attention.self.query.weight is different
# key module.text_branch.encoder.layer.2.attention.self.query.bias is different
# key module.text_branch.encoder.layer.2.attention.self.key.weight is different
# key module.text_branch.encoder.layer.2.attention.self.key.bias is different
# key module.text_branch.encoder.layer.2.attention.self.value.weight is different
# key module.text_branch.encoder.layer.2.attention.self.value.bias is different
# key module.text_branch.encoder.layer.2.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.2.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.2.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.2.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.2.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.2.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.2.output.dense.weight is different
# key module.text_branch.encoder.layer.2.output.dense.bias is different
# key module.text_branch.encoder.layer.2.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.2.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.3.attention.self.query.weight is different
# key module.text_branch.encoder.layer.3.attention.self.query.bias is different
# key module.text_branch.encoder.layer.3.attention.self.key.weight is different
# key module.text_branch.encoder.layer.3.attention.self.key.bias is different
# key module.text_branch.encoder.layer.3.attention.self.value.weight is different
# key module.text_branch.encoder.layer.3.attention.self.value.bias is different
# key module.text_branch.encoder.layer.3.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.3.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.3.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.3.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.3.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.3.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.3.output.dense.weight is different
# key module.text_branch.encoder.layer.3.output.dense.bias is different
# key module.text_branch.encoder.layer.3.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.3.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.4.attention.self.query.weight is different
# key module.text_branch.encoder.layer.4.attention.self.query.bias is different
# key module.text_branch.encoder.layer.4.attention.self.key.weight is different
# key module.text_branch.encoder.layer.4.attention.self.key.bias is different
# key module.text_branch.encoder.layer.4.attention.self.value.weight is different
# key module.text_branch.encoder.layer.4.attention.self.value.bias is different
# key module.text_branch.encoder.layer.4.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.4.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.4.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.4.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.4.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.4.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.4.output.dense.weight is different
# key module.text_branch.encoder.layer.4.output.dense.bias is different
# key module.text_branch.encoder.layer.4.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.4.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.5.attention.self.query.weight is different
# key module.text_branch.encoder.layer.5.attention.self.query.bias is different
# key module.text_branch.encoder.layer.5.attention.self.key.weight is different
# key module.text_branch.encoder.layer.5.attention.self.key.bias is different
# key module.text_branch.encoder.layer.5.attention.self.value.weight is different
# key module.text_branch.encoder.layer.5.attention.self.value.bias is different
# key module.text_branch.encoder.layer.5.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.5.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.5.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.5.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.5.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.5.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.5.output.dense.weight is different
# key module.text_branch.encoder.layer.5.output.dense.bias is different
# key module.text_branch.encoder.layer.5.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.5.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.6.attention.self.query.weight is different
# key module.text_branch.encoder.layer.6.attention.self.query.bias is different
# key module.text_branch.encoder.layer.6.attention.self.key.weight is different
# key module.text_branch.encoder.layer.6.attention.self.key.bias is different
# key module.text_branch.encoder.layer.6.attention.self.value.weight is different
# key module.text_branch.encoder.layer.6.attention.self.value.bias is different
# key module.text_branch.encoder.layer.6.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.6.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.6.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.6.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.6.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.6.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.6.output.dense.weight is different
# key module.text_branch.encoder.layer.6.output.dense.bias is different
# key module.text_branch.encoder.layer.6.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.6.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.7.attention.self.query.weight is different
# key module.text_branch.encoder.layer.7.attention.self.query.bias is different
# key module.text_branch.encoder.layer.7.attention.self.key.weight is different
# key module.text_branch.encoder.layer.7.attention.self.key.bias is different
# key module.text_branch.encoder.layer.7.attention.self.value.weight is different
# key module.text_branch.encoder.layer.7.attention.self.value.bias is different
# key module.text_branch.encoder.layer.7.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.7.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.7.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.7.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.7.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.7.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.7.output.dense.weight is different
# key module.text_branch.encoder.layer.7.output.dense.bias is different
# key module.text_branch.encoder.layer.7.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.7.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.8.attention.self.query.weight is different
# key module.text_branch.encoder.layer.8.attention.self.query.bias is different
# key module.text_branch.encoder.layer.8.attention.self.key.weight is different
# key module.text_branch.encoder.layer.8.attention.self.key.bias is different
# key module.text_branch.encoder.layer.8.attention.self.value.weight is different
# key module.text_branch.encoder.layer.8.attention.self.value.bias is different
# key module.text_branch.encoder.layer.8.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.8.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.8.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.8.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.8.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.8.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.8.output.dense.weight is different
# key module.text_branch.encoder.layer.8.output.dense.bias is different
# key module.text_branch.encoder.layer.8.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.8.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.9.attention.self.query.weight is different
# key module.text_branch.encoder.layer.9.attention.self.query.bias is different
# key module.text_branch.encoder.layer.9.attention.self.key.weight is different
# key module.text_branch.encoder.layer.9.attention.self.key.bias is different
# key module.text_branch.encoder.layer.9.attention.self.value.weight is different
# key module.text_branch.encoder.layer.9.attention.self.value.bias is different
# key module.text_branch.encoder.layer.9.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.9.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.9.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.9.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.9.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.9.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.9.output.dense.weight is different
# key module.text_branch.encoder.layer.9.output.dense.bias is different
# key module.text_branch.encoder.layer.9.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.9.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.10.attention.self.query.weight is different
# key module.text_branch.encoder.layer.10.attention.self.query.bias is different
# key module.text_branch.encoder.layer.10.attention.self.key.weight is different
# key module.text_branch.encoder.layer.10.attention.self.key.bias is different
# key module.text_branch.encoder.layer.10.attention.self.value.weight is different
# key module.text_branch.encoder.layer.10.attention.self.value.bias is different
# key module.text_branch.encoder.layer.10.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.10.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.10.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.10.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.10.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.10.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.10.output.dense.weight is different
# key module.text_branch.encoder.layer.10.output.dense.bias is different
# key module.text_branch.encoder.layer.10.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.10.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.11.attention.self.query.weight is different
# key module.text_branch.encoder.layer.11.attention.self.query.bias is different
# key module.text_branch.encoder.layer.11.attention.self.key.weight is different
# key module.text_branch.encoder.layer.11.attention.self.key.bias is different
# key module.text_branch.encoder.layer.11.attention.self.value.weight is different
# key module.text_branch.encoder.layer.11.attention.self.value.bias is different
# key module.text_branch.encoder.layer.11.attention.output.dense.weight is different
# key module.text_branch.encoder.layer.11.attention.output.dense.bias is different
# key module.text_branch.encoder.layer.11.attention.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.11.attention.output.LayerNorm.bias is different
# key module.text_branch.encoder.layer.11.intermediate.dense.weight is different
# key module.text_branch.encoder.layer.11.intermediate.dense.bias is different
# key module.text_branch.encoder.layer.11.output.dense.weight is different
# key module.text_branch.encoder.layer.11.output.dense.bias is different
# key module.text_branch.encoder.layer.11.output.LayerNorm.weight is different
# key module.text_branch.encoder.layer.11.output.LayerNorm.bias is different
# a_sum:  tensor(15078.6641)
# b_sum:  tensor(15540.0723)
# diff:  tensor(-461.4082)
# True

# linear_prob_text
# check_ckpt_diff("/fsx/clap_logs/2022_09_15-02_05_29-linear_probemodel_PANN-14-lr_0.0001-b_512-j_4-p_fp32/checkpoints/pretrain_epoch_10_lp_epoch_50.pt", "/fsx/clap_logs/2022_09_15-02_05_29-linear_probemodel_PANN-14-lr_0.0001-b_512-j_4-p_fp32/checkpoints/pretrain_epoch_10_lp_epoch_100.pt", "text_branch.resblocks")

# a_sum:  tensor(12111.0244)
# b_sum:  tensor(12111.0244)
# diff:  tensor(0.)

# linear_prob_audio
# check_ckpt_diff("/fsx/clap_logs/2022_09_15-02_05_29-linear_probemodel_PANN-14-lr_0.0001-b_512-j_4-p_fp32/checkpoints/pretrain_epoch_10_lp_epoch_50.pt", "/fsx/clap_logs/2022_09_15-02_05_29-linear_probemodel_PANN-14-lr_0.0001-b_512-j_4-p_fp32/checkpoints/pretrain_epoch_10_lp_epoch_100.pt", "clap_model")

# key clap_model.audio_branch.bn0.num_batches_tracked is different
# key clap_model.audio_branch.conv_block1.bn1.running_mean is different
# key clap_model.audio_branch.conv_block1.bn1.running_var is different
# key clap_model.audio_branch.conv_block1.bn1.num_batches_tracked is different
# key clap_model.audio_branch.conv_block1.bn2.running_mean is different
# key clap_model.audio_branch.conv_block1.bn2.running_var is different
# key clap_model.audio_branch.conv_block1.bn2.num_batches_tracked is different
# key clap_model.audio_branch.conv_block2.bn1.running_mean is different
# key clap_model.audio_branch.conv_block2.bn1.running_var is different
# key clap_model.audio_branch.conv_block2.bn1.num_batches_tracked is different
# key clap_model.audio_branch.conv_block2.bn2.running_mean is different
# key clap_model.audio_branch.conv_block2.bn2.running_var is different
# key clap_model.audio_branch.conv_block2.bn2.num_batches_tracked is different
# key clap_model.audio_branch.conv_block3.bn1.running_mean is different
# key clap_model.audio_branch.conv_block3.bn1.running_var is different
# key clap_model.audio_branch.conv_block3.bn1.num_batches_tracked is different
# key clap_model.audio_branch.conv_block3.bn2.running_mean is different
# key clap_model.audio_branch.conv_block3.bn2.running_var is different
# key clap_model.audio_branch.conv_block3.bn2.num_batches_tracked is different
# key clap_model.audio_branch.conv_block4.bn1.running_mean is different
# key clap_model.audio_branch.conv_block4.bn1.running_var is different
# key clap_model.audio_branch.conv_block4.bn1.num_batches_tracked is different
# key clap_model.audio_branch.conv_block4.bn2.running_mean is different
# key clap_model.audio_branch.conv_block4.bn2.running_var is different
# key clap_model.audio_branch.conv_block4.bn2.num_batches_tracked is different
# key clap_model.audio_branch.conv_block5.bn1.running_mean is different
# key clap_model.audio_branch.conv_block5.bn1.running_var is different
# key clap_model.audio_branch.conv_block5.bn1.num_batches_tracked is different
# key clap_model.audio_branch.conv_block5.bn2.running_mean is different
# key clap_model.audio_branch.conv_block5.bn2.running_var is different
# key clap_model.audio_branch.conv_block5.bn2.num_batches_tracked is different
# key clap_model.audio_branch.conv_block6.bn1.running_mean is different
# key clap_model.audio_branch.conv_block6.bn1.running_var is different
# key clap_model.audio_branch.conv_block6.bn1.num_batches_tracked is different
# key clap_model.audio_branch.conv_block6.bn2.running_mean is different
# key clap_model.audio_branch.conv_block6.bn2.running_var is different
# key clap_model.audio_branch.conv_block6.bn2.num_batches_tracked is different
# a_sum:  tensor(120061.5078)
# b_sum:  tensor(122656.0469)
# diff:  tensor(-2594.5391)
# True

