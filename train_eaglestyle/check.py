from safetensors import safe_open

# 替换为你的 .safetensors 文件路径
file_path = "/share/wanghanzhen/MTP/dMTP/train_eaglestyle/mtp_checkpoints_whead/model-00001-of-00002.safetensors"

with safe_open(file_path, framework="pt", device="cpu") as f:
    # 获取所有张量名称
    tensor_names = f.keys()
    for name in sorted(tensor_names):
        print(f"{name}")